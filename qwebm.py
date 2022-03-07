from ast import arg
from math import floor
from os import path
import os
import subprocess
import asyncio
import json
import re
from operator import itemgetter
import logging
import argparse

DIM_MULT = 4  # the number the dimensions should be divisible by; improves playback compatibility
MD_LONG = 960  # medium size video dimension of the long side (usually width)
MD_TARGET_SIZE_KB = 4 * 1024  # medium size video target size
MD_WITH_AUDIO_TARGET_SIZE_KB = 6 * 1024 # medium size video w/ audio target size
MD_CRF = 9

MAX_PASS = 5

logger = logging.getLogger("qwebm")


def probe_file(path):
    result = subprocess.run(
        [
            "ffprobe",
            "-hide_banner",
            "-loglevel",
            "error",
            "-show_streams",
            "-print_format",
            "json",
            path,
        ],
        capture_output=True,
    )
    # print(result)
    json_obj = json.loads(result.stdout)
    print(json.dumps(json_obj, indent=2))

    print("codec name: " + json_obj["streams"][0]["codec_name"])
    return json_obj["streams"][0]


def _get_path_breakdown(_path):
    basename = path.basename(_path)
    dirname = path.dirname(_path)
    dot_idx = basename.rfind(".") if basename.rfind(".") >= 0 else len(basename)
    return basename, dirname, dot_idx


def resolve_file_name_conflict(source_file_path, target_file_path):
    if not path.exists(target_file_path):
        return target_file_path

    s_bn = path.basename(source_file_path)
    t_bn, t_dn, t_dot_idx = _get_path_breakdown(target_file_path)

    def _get_basename_parts(_basename):
        final_dot_idx = _basename.rfind(".")
        return _basename[0:final_dot_idx], _basename[final_dot_idx + 1 :]

    s_bn_before_ext, _ = _get_basename_parts(s_bn)
    t_bn_before_ext, t_bn_ext = _get_basename_parts(t_bn)

    def _has_dash_numeric_postfix(_bn_before_ext):
        dash_idx = _bn_before_ext.rfind("-")
        after_dash = _bn_before_ext[dash_idx + 1 :]
        return after_dash.isnumeric()

    def _generate_new_path(fnum):
        dash_idx = t_bn_before_ext.rfind("-")

        new_bn_before_ext = ""
        if dash_idx >= 0 and not _has_dash_numeric_postfix(s_bn_before_ext):
            _bname_before_dash = t_bn_before_ext[0:dash_idx]
            _bname_after_dash = t_bn_before_ext[dash_idx + 1 :]
            if _bname_after_dash.isnumeric():
                new_bn_before_ext = f"{_bname_before_dash}-{fnum}"

        if new_bn_before_ext == "":
            new_bn_before_ext = f"{t_bn_before_ext}-{fnum}"

        new_basename = (
            f"{new_bn_before_ext}.{t_bn_ext}" if t_dot_idx >= 0 else new_bn_before_ext
        )
        return path.join(t_dn, new_basename)

    l_num = 1
    while True:
        new_path = _generate_new_path(l_num)
        if not path.exists(new_path):
            return new_path
        l_num += 1


def generate_target_file_path(source_path):
    # basename = path.basename(source_path)
    # dirname = path.dirname(source_path)
    # dot_idx = basename.rfind(".") if basename.rfind(".") >= 0 else len(basename)
    basename, dirname, dot_idx = _get_path_breakdown(source_path)
    target_basename = basename[0:dot_idx] + ".webm"
    # return path.join(dirname, target_basename)
    tentative_target_path = path.join(dirname, target_basename)
    return resolve_file_name_conflict(source_path, tentative_target_path)


def generate_ffmpeg_options(
    video_info,
    preset="medium",
    video_bitrate=None,
    video_bitrate_mult=1.0,
    crf_adjust=0,
):

    [width, height] = map(int, itemgetter("width", "height")(video_info))
    codec_name = video_info["codec_name"]
    duration = float(video_info["duration"])

    n_width, n_height = [width, height]

    is_vertical = height > width
    if max(width, height) > MD_LONG * 1.2:
        if not is_vertical:
            n_width = MD_LONG
            n_height = round((height / width * n_width) / DIM_MULT) * DIM_MULT
        else:
            n_height = MD_LONG
            n_width = round((width / height * n_height) / DIM_MULT) * DIM_MULT
    else:
        n_width = round(width / DIM_MULT) * DIM_MULT
        n_height = round(height / DIM_MULT) * DIM_MULT

    options = [
        "-c:v",
        "libvpx",
        "-an",
        "-sn",
        "-dn",
        "-map_metadata",
        "-1",
    ]
    if n_width != width or n_height != height:
        options.append("-vf")
        options.append(f"scale={n_width}:{n_height}")

    target_bitrate = video_bitrate or floor(
        MD_TARGET_SIZE_KB / duration * 8 * 1000 * video_bitrate_mult
    )
    options.append("-b:v")
    options.append(f"{target_bitrate}")

    options.append("-crf")
    options.append(f"{MD_CRF + crf_adjust}")

    if codec_name == "gif":
        options.append("-auto-alt-ref")
        options.append("0")

    return options


def transcode_file(source_path, preset="medium"):
    v0info = probe_file(source_path)
    # relevant: width, height, codec_name, duration

    options = generate_ffmpeg_options(v0info)

    def generate_run_args(source_path, ffmpeg_options):
        return [
            "ffmpeg",
            "-i",
            f"{source_path}",
            *ffmpeg_options,
            f"{generate_target_file_path(source_path)}",
        ]

    run_args = generate_run_args(source_path, options)

    # result = subprocess.run(run_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # result = subprocess.Popen(run_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # for line in iter(result.stderr.readline, b''):
    #     print(f"err> {line}")

    # for line in iter(result.stdout.readline, b''):
    #     print(f"out> {line}")

    # print(result)
    async def run_ffmpeg(ffmpeg_args):
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg",
            *ffmpeg_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        progress_frame_fps_pattern = re.compile("frame=\s*(\d+) fps=\s*(\d+)")
        progress_size_time_pattern = re.compile(
            "size=\s*(\d+)kB time=(\d{2}:\d{2}:\d{2}.\d{2})"
        )
        summary_pattern = re.compile(
            "video:(\d+)kB audio:(\d+)kB subtitle:(\d+)kB other streams:(\d+)kB "
            "global headers:(\d+)kB muxing overhead: (\d+\.\d+)%"
        )
        err_buf = bytearray()

        total_kb = (
            video_kb
        ) = (
            audio_kb
        ) = subtitle_kb = other_streams_kb = global_headers_kb = mux_overhead_pct = 0
        progress_timecode = None
        progress_pct = 0
        while not proc.stderr.at_eof():
            err_byte = await proc.stderr.read(1)
            err_buf += err_byte
            if err_byte.decode() == "\r":
                err_str = err_buf.decode()
                err_buf.clear()
                p_ff_matches = progress_frame_fps_pattern.findall(err_str)
                p_st_matches = progress_size_time_pattern.findall(err_str)

                if p_ff_matches:
                    progress_pct = round(
                        int(p_ff_matches[0][0]) / int(v0info["nb_frames"]) * 100, 1
                    )

                if p_st_matches:
                    # print(f"matches: {p_st_matches[0][0]} kB, {p_st_matches[0][1]}")
                    total_kb = int(p_st_matches[0][0])
                    progress_timecode = p_st_matches[0][1]

                    print(
                        f"\r * {progress_pct:.2f}% done; file up to {total_kb} kB; "
                        f"video time up to {progress_timecode} ",
                        end="",
                    )
                else:
                    print("")
                    s_matches = summary_pattern.findall(err_str)
                    if s_matches:
                        print(f"summary: {s_matches}")
                        (
                            video_kb,
                            audio_kb,
                            subtitle_kb,
                            other_streams_kb,
                            global_headers_kb,
                        ) = map(int, s_matches[0][:5])
                        mux_overhead_pct = float(s_matches[0][5])

        await proc.wait()
        return {
            "output_path": run_args[-1],
            "total_kb": total_kb,
            "progress_timecode": progress_timecode,
            "progress_pct": progress_pct,
            "video_kb": video_kb,
            "audio_kb": audio_kb,
            "subtitle_kb": subtitle_kb,
            "other_streams_kb": other_streams_kb,
            "global_headers_kb": global_headers_kb,
            "mux_overhead_pct": mux_overhead_pct,
        }

    output_size_kb = MD_TARGET_SIZE_KB + 1
    pass_num = 0
    while pass_num < MAX_PASS and output_size_kb >= MD_TARGET_SIZE_KB:
        print(f"Pass {pass_num}:")
        result = asyncio.run(run_ffmpeg(run_args[1:]))
        print(f"run_ffmpeg result: {result}")
        output_size_kb = result["total_kb"]
        output_video_kb = result["video_kb"]

        output_path = result["output_path"]
        if output_size_kb >= MD_TARGET_SIZE_KB:
            try:
                os.remove(output_path)
            except:
                logger.info(
                    f"Could not delete {output_path}, "
                    "output that failed to meet size requirement."
                )

        old_video_bit_rate = run_args[run_args.index("-b:v") + 1]

        new_video_bitrate = round(
            int(old_video_bit_rate)
            * (output_video_kb - (output_size_kb - MD_TARGET_SIZE_KB))
            / output_video_kb
        )

        run_args = generate_run_args(
            source_path,
            generate_ffmpeg_options(
                v0info,
                video_bitrate=new_video_bitrate,
                crf_adjust=floor(pass_num / 2),
            ),
        )

        print(f"new run_args: {run_args}")

        pass_num += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcodes video to VP8 WebM. "
        "Makes sure the resulting file is under a specified size."
    )

    parser.add_argument("-a", "--audio", action="store_true", help="include audio")
    parser.add_argument("-s", "--size", action="store", type=int, help="target size in kB")
    parser.add_argument("input_video_file", nargs=1)
    args = parser.parse_args()
    print(args)
    print(args.size)

    # transcode_file("")
