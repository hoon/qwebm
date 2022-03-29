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
import tempfile
import sys
import time

DIM_MULT = 4  # the number the dimensions should be divisible by; improves playback compatibility
MD_LONG = 960  # medium size video dimension of the long side (usually width)
MD_TARGET_SIZE_KB = 3 * 1024  # medium size video target size
MD_WITH_AUDIO_TARGET_SIZE_KB = 6 * 1024  # medium size video w/ audio target size
MD_CRF = 9
MD_AUDIO_QSCALE = 3

MAX_TRY = 5
MAX_BIT_RATE_MULT = 4

logger = logging.getLogger("qwebm")


def probe_file(path):
    result = subprocess.run(
        [
            "ffprobe",
            "-hide_banner",
            "-loglevel",
            "error",
            "-show_streams",
            "-show_format",
            "-print_format",
            "json",
            path,
        ],
        capture_output=True,
    )
    logger.debug(f"ffprobe output raw: {result}")
    json_obj = json.loads(result.stdout)
    logger.debug(f"ffprobe formatted json: {json.dumps(json_obj, indent=2)}")

    return json_obj


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


def get_vorbis_target_bitrate_kbps(qscale):
    if qscale < 4:
        return 16 * (qscale + 4)
    elif qscale < 8:
        return 32 * qscale
    else:
        return 64 * (qscale - 4)


def get_target_video_dimension(source_width, source_height):
    n_width, n_height = [source_width, source_height]

    is_vertical = source_height > source_width
    if max(source_width, source_height) > MD_LONG * 1.2:
        if not is_vertical:
            n_width = MD_LONG
            n_height = (
                round((source_height / source_width * n_width) / DIM_MULT) * DIM_MULT
            )
        else:
            n_height = MD_LONG
            n_width = (
                round((source_width / source_height * n_height) / DIM_MULT) * DIM_MULT
            )
    else:
        n_width = round(source_width / DIM_MULT) * DIM_MULT
        n_height = round(source_height / DIM_MULT) * DIM_MULT

    return n_width, n_height


def generate_ffmpeg_options(
    video_info,
    audio_info=None,
    file_format_info=None,
    include_audio=False,
    audio_qscale_adjust=0,
    target_size_kb=None,
    preset="medium",
    video_bitrate=None,
    video_bitrate_mult=1.0,
    crf_adjust=0,
    pass_no=None,
    source_path=None,
    target_path=None,
):

    [width, height] = map(int, itemgetter("width", "height")(video_info))
    codec_name = video_info["codec_name"]
    duration = float(
        video_info["duration"]
        if "duration" in video_info
        else file_format_info["duration"]
    )

    n_width, n_height = get_target_video_dimension(width, height)

    options = []

    new_target_path = target_path

    if source_path:
        options.append("-i")
        options.append(source_path)

        if not target_path:
            new_target_path = generate_target_file_path(source_path)

    options.append("-c:v")
    options.append("libvpx")
    options.append("-sn")
    options.append("-dn")

    if n_width != width or n_height != height:
        options.append("-vf")
        options.append(f"scale={n_width}:{n_height}")

    target_size_kb = MD_TARGET_SIZE_KB if not target_size_kb else target_size_kb

    target_video_bitrate = None

    if video_bitrate:
        target_video_bitrate = video_bitrate
    else:
        target_video_size = target_size_kb
        if include_audio:
            target_audio_bitrate_kbps = get_vorbis_target_bitrate_kbps(
                MD_AUDIO_QSCALE + audio_qscale_adjust
            )
            target_video_size = target_size_kb - (
                target_audio_bitrate_kbps * duration / 8
            )

        target_video_bitrate = floor(
            target_video_size / duration * 8 * 1000 * video_bitrate_mult
        )

    options.append("-b:v")
    options.append(f"{target_video_bitrate}")

    options.append("-maxrate")
    options.append(f"{target_video_bitrate * MAX_BIT_RATE_MULT}")

    options.append("-crf")
    options.append(f"{MD_CRF + crf_adjust}")

    if include_audio:
        options.append("-c:a")
        options.append("libvorbis")
        options.append("-qscale:a")
        options.append(f"{MD_AUDIO_QSCALE + audio_qscale_adjust}")
    else:
        options.append("-an")

    if codec_name == "gif":
        options.append("-auto-alt-ref")
        options.append("0")

    if pass_no == 1:
        options.append("-pass")
        options.append("1")
        options.append("-f")
        options.append("null")
        if sys.platform == "win32":
            options.append("NUL")
        else:
            options.append("/dev/null")
    elif pass_no == 2:
        options.append("-pass")
        options.append("2")
        if new_target_path:
            options.append(new_target_path)

    return options


def get_stream(media_info, codec_type):
    if not media_info or "streams" not in media_info:
        return None

    for m_stream in media_info["streams"]:
        if m_stream["codec_type"] == codec_type:
            return m_stream

    return None


def convert_timestamp_to_sec(timestamp_str):
    timestamp_pattern = re.compile("(\d{2}):(\d{2}):(\d{2}).(\d+)")
    ts_matches = timestamp_pattern.findall(timestamp_str)

    if len(ts_matches) < 1:
        return None
    else:
        return (
            int(ts_matches[0][0]) * 60 * 60
            + int(ts_matches[0][1]) * 60
            + int(ts_matches[0][2])
            + int(ts_matches[0][3]) / (10 ** (len(ts_matches[0][3]) + 1))
        )


def generate_time_wheel(nanosec):
    _gtw_SHIFT_EVERY_MS = 80
    wheel_chars = ["|", "/", "-", "\\"]
    return wheel_chars[floor(nanosec / 1000 / _gtw_SHIFT_EVERY_MS) % len(wheel_chars)]


def compute_aux_info(media_info):
    aux_info = dict()
    video_info = get_stream(media_info, "video")
    duration_str = None
    if "tags" in video_info and "DURATION" in video_info["tags"]:
        duration_str = video_info["tags"]["DURATION"]
    elif "format" in media_info and "duration" in media_info["format"]:
        duration_str = media_info["format"]["duration"]

    if duration_str:
        aux_info["duration_sec"] = convert_timestamp_to_sec(duration_str)

    return aux_info


async def run_ffmpeg(video_info, ffmpeg_args, file_format_info=None, aux_info=None):
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        *ffmpeg_args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    duration_sec = aux_info["duration_sec"] if "duration_sec" in aux_info else None

    # example ffmpeg outputs to match:
    # frame=  442 fps= 95 q=0.0 Lsize=    1049kB time=00:00:14.70 bitrate= 584.4kbits/s speed=3.17x
    # video:1045kB audio:0kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 0.370684%
    # example pass 1
    # frame= 1184 fps=185 q=0.0 Lsize=N/A time=00:00:39.49 bitrate=N/A speed=6.18x
    progress_frame_fps_pattern = re.compile("frame=\s*(\d+) fps=\s*(\d+)")
    progress_size_pattern = re.compile("size=\s*(\d+)kB")
    progress_time_pattern = re.compile("time=(\d{2}:\d{2}:\d{2}.\d{2})")
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
    progress_end_newline_printed = False
    start_time_ns = time.time_ns()
    while not proc.stderr.at_eof():
        err_byte = await proc.stderr.read(1)
        err_buf += err_byte
        err_byte_decoded = None
        try:
            # in case we encounter multi-byte character in utf-8
            err_byte_decoded = err_byte.decode()
        except:
            pass

        if err_byte_decoded == "\r" or err_byte_decoded == "\n":
            err_str = None
            try:
                # in case the buffer can't be decoded to utf-8
                err_str = err_buf.decode()
            except:
                pass
            finally:
                err_buf.clear()

            if not err_str:
                continue

            logger.debug(err_str)

            p_ff_matches = progress_frame_fps_pattern.findall(err_str)
            p_size_matches = progress_size_pattern.findall(err_str)
            p_time_matches = progress_time_pattern.findall(err_str)

            # 'Lsize=' is printed at the final progress update output
            if err_str.find("Lsize=") >= 0:
                progress_pct = 100

            if p_ff_matches:
                if "nb_frames" in video_info:
                    progress_pct = max(
                        round(
                            int(p_ff_matches[0][0])
                            / int(video_info["nb_frames"])
                            * 100,
                            1,
                        ),
                        progress_pct,
                    )

                if p_size_matches:
                    logger.debug(f"ffmpeg output matches: {p_size_matches[0][0]} kB")
                    total_kb = int(p_size_matches[0])

                if p_time_matches:
                    progress_timecode = p_time_matches[0]
                    logger.debug(f"ffmpeg output timecode match: {progress_timecode}")

                    current_timecode_sec = convert_timestamp_to_sec(progress_timecode)

                    if (
                        "nb_frames" not in video_info
                        and duration_sec is not None
                        and duration_sec > 0
                    ):
                        progress_pct = max(
                            round(
                                current_timecode_sec / duration_sec * 100,
                                1,
                            ),
                            progress_pct,
                        )

                    if progress_pct > 0 and current_timecode_sec > 0:
                        progress_str = f"\r * {progress_pct:.2f}% done"
                        if total_kb > 0:
                            progress_str += f"; file up to {total_kb} kB"

                        if progress_timecode != "00:00:00.00":
                            progress_str += f"; video time up to {progress_timecode}"

                        progress_str += " "

                        print(progress_str, end="")
                    else:
                        print(f"\r * {progress_pct:.2f}% done ", end="")

                elif progress_pct > 0:
                    print(f"\r * {progress_pct:.2f}% done ", end="")

                if progress_pct <= 0:
                    print(
                        f"\r {generate_time_wheel(time.time_ns() - start_time_ns)} Processing...",
                        end="",
                    )

            else:
                logger.debug(f"progress output pattern not matched:\n{err_str}")
                s_matches = summary_pattern.findall(err_str)

                if not progress_end_newline_printed:
                    if len(s_matches) > 0:
                        progress_end_newline_printed = True
                        print("")
                    # pass 1 doesn't output mux overhead, so summary_pattern doesn't match
                    elif err_str.find("video:0kB") > -1:
                        # if percent progress was able to be displayed for pass 1
                        if "nb_frames" in video_info:
                            print("")
                        # else clear "Processing..." and print "Done"
                        else:
                            print("\r" + " " * 20, end="")
                            print("\r * Done")

                if s_matches:
                    (
                        video_kb,
                        audio_kb,
                        subtitle_kb,
                        other_streams_kb,
                        global_headers_kb,
                    ) = map(int, s_matches[0][:5])
                    mux_overhead_pct = float(s_matches[0][5])
                    print(
                        f"Result: video {video_kb} kB, audio {audio_kb} kB, "
                        f"global headers {global_headers_kb} kB, "
                        f"mux overhead {(mux_overhead_pct):.2f}%"
                    )

    await proc.wait()
    return {
        "output_path": ffmpeg_args[-1],
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


def two_pass_transcode_file(
    source_path,
    video_info=None,
    audio_info=None,
    file_format_info=None,
    aux_info=None,
    include_audio=False,
    target_size=None,
    preset="medium",
):
    output_size_kb = MD_TARGET_SIZE_KB + 1
    try_no = 1
    ffmpeg_options_adjustments = {
        "video_bitrate": None,
        "crf_adjust": 0,
        "audio_qscale_adjust": 0,
    }
    while try_no <= MAX_TRY and output_size_kb >= MD_TARGET_SIZE_KB:

        ffmpeg_options_pass_1 = generate_ffmpeg_options(
            video_info,
            source_path=source_path,
            audio_info=audio_info,
            file_format_info=file_format_info,
            include_audio=include_audio,
            target_size_kb=target_size,
            pass_no=1,
            preset=preset,
            video_bitrate=ffmpeg_options_adjustments["video_bitrate"],
            crf_adjust=ffmpeg_options_adjustments["crf_adjust"],
            audio_qscale_adjust=ffmpeg_options_adjustments["audio_qscale_adjust"],
        )

        print(f"Try {try_no}:")
        print(f"Pass 1 of 2:")
        logger.info(f"Options: {ffmpeg_options_pass_1}")
        pass_1_result = asyncio.run(
            run_ffmpeg(
                video_info,
                ffmpeg_options_pass_1,
                file_format_info=file_format_info,
                aux_info=aux_info,
            )
        )
        logger.info(f"pass 1 result: {pass_1_result}")

        print(f"Pass 2 of 2:")
        ffmpeg_options_pass_2 = generate_ffmpeg_options(
            video_info,
            source_path=source_path,
            audio_info=audio_info,
            file_format_info=file_format_info,
            include_audio=include_audio,
            target_size_kb=target_size,
            pass_no=2,
            preset=preset,
            video_bitrate=ffmpeg_options_adjustments["video_bitrate"],
            crf_adjust=ffmpeg_options_adjustments["crf_adjust"],
            audio_qscale_adjust=ffmpeg_options_adjustments["audio_qscale_adjust"],
        )
        logger.info(f"Options: {ffmpeg_options_pass_2}")
        pass_2_result = asyncio.run(
            run_ffmpeg(
                video_info,
                ffmpeg_options_pass_2,
                file_format_info=file_format_info,
                aux_info=aux_info,
            )
        )

        logger.info(f"\npass 2 result: {pass_2_result}")

        output_size_kb = pass_2_result["total_kb"]
        output_video_kb = pass_2_result["video_kb"]

        output_path = pass_2_result["output_path"]
        if output_size_kb >= MD_TARGET_SIZE_KB:
            try:
                os.remove(output_path)
            except:
                logger.warning(
                    f"Could not delete {output_path}, "
                    "output that failed to meet size requirement."
                )

        old_video_bit_rate = ffmpeg_options_pass_2[
            ffmpeg_options_pass_2.index("-b:v") + 1
        ]

        new_video_bitrate = round(
            int(old_video_bit_rate)
            * (output_video_kb - (output_size_kb - MD_TARGET_SIZE_KB))
            / output_video_kb
        )

        ffmpeg_options_adjustments = {
            "video_bitrate": new_video_bitrate,
            "crf_adjust": floor(try_no / 2),
            "audio_qscale_adjust": -try_no / 4,
        }

        try_no += 1


def two_pass_transcode(input_path, include_audio=False, size=None):
    media_info = probe_file(input_path)
    if "streams" not in media_info or "format" not in media_info:
        print(
            f"ffprobe failed to read information from the file '{input_path}'. "
            "Please check if the file is a valid video file."
        )
        return None
    video_info = get_stream(media_info, "video")
    audio_info = get_stream(media_info, "audio")
    file_format_info = media_info["format"]
    aux_info = compute_aux_info(media_info)

    with tempfile.TemporaryDirectory() as temp_dir:
        two_pass_transcode_file(
            input_path,
            video_info,
            audio_info=audio_info,
            file_format_info=file_format_info,
            aux_info=aux_info,
            include_audio=include_audio,
        )

    return None


def set_log_level(log_level):
    lvl_str = str(log_level).strip().upper()

    if lvl_str in ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]:
        lsh = logging.StreamHandler()
        lsh.setLevel(lvl_str)
        logger.addHandler(lsh)
        logger.setLevel(lvl_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcodes video to VP8 WebM. "
        "Makes sure the resulting file is under a specified size."
    )

    parser.add_argument(
        "--loglevel",
        default="warning",
        help="set log level (debug, info, warning, error, critical), defaults to warning",
    )

    parser.add_argument("-a", "--audio", action="store_true", help="include audio")
    parser.add_argument(
        "-s", "--size", action="store", type=int, help="target size in kB"
    )
    parser.add_argument("input_video_file")
    args = parser.parse_args()

    set_log_level(args.loglevel)

    logger.info(f"Command line arguments: {args}")

    if not path.exists(args.input_video_file):
        print(
            f"Specified file does not exist: {args.input_video_file}", file=sys.stderr
        )
        exit(-1)

    two_pass_transcode(args.input_video_file, include_audio=args.audio, size=args.size)
