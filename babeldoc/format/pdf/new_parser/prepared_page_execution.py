from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from babeldoc.format.pdf.new_parser.bridge_types import EmitterSink
from babeldoc.format.pdf.new_parser.native_page_interpreter import _emit_page_to_sink
from babeldoc.format.pdf.new_parser.native_page_interpreter import _parse_page_heavy
from babeldoc.format.pdf.new_parser.page_interpreter import PageInterpreter
from babeldoc.format.pdf.new_parser.prepared_page import PreparedPdfPage
from babeldoc.format.pdf.new_parser.resource_runtime_types import PageResourceRuntime


def run_prepared_pages(
    sink: EmitterSink,
    should_translate_page: Callable[[int], bool],
    selected_pages: list[PreparedPdfPage],
    page_interpreter: PageInterpreter,
    resource_runtime: PageResourceRuntime | None = None,
    text_run_positioner: object = None,
) -> object:
    """Run the active parser across *selected_pages*.

    The heavy lift — parsing each page's PDF content stream with PyMuPDF
    (``interpret_page_with_resource_bundle``) — releases the GIL, so we
    execute it in parallel via ``ThreadPoolExecutor``.  The lighter sink
    emission (``emit_native_text_events_to_legacy_sink``) is GIL-bound and
    must run serially in page order to keep ``sink`` state consistent.
    """
    pages_to_translate: list[PreparedPdfPage] = [
        p for p in selected_pages if should_translate_page(p.pageno + 1)
    ]
    total_pages = len(pages_to_translate)
    sink.on_total_pages(total_pages)

    # ---- serial fallback (old code path) -----------------------------------
    if resource_runtime is None or text_run_positioner is None:
        for page in pages_to_translate:
            page_interpreter.begin_page(page, page.pageno)
            ops_base = page_interpreter.process_page(page)
            sink.on_page_base_operation(ops_base)
            sink.on_page_end()
            page_interpreter.end_page(page, page.pageno)
        sink.on_finish()
        return sink.create_il()

    # ---- phase 1: heavy parse (GIL-released → parallel) -------------------
    n_workers = min(max((os.cpu_count() or 4) // 2, 1), total_pages, 16)
    parsed: dict[int, tuple] = {}

    if n_workers > 1 and total_pages > 1:
        def _parse(page: PreparedPdfPage):
            return page.pageno, _parse_page_heavy(page, resource_runtime)

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_parse, page): page.pageno
                for page in pages_to_translate
            }
            for future in as_completed(futures):
                pageno, heavy_result = future.result()
                parsed[pageno] = heavy_result
    else:
        for page in pages_to_translate:
            parsed[page.pageno] = _parse_page_heavy(page, resource_runtime)

    # ---- phase 2: sink emission (GIL-bound → serial, page order) ----------
    for page in pages_to_translate:
        events, resource_bundle, base_operations = parsed[page.pageno]
        # Replay the begin_page side-effects now (they were skipped by the
        # parallel phase because they write into the shared sink).
        x0, y0, x1, y1 = page.cropbox
        sink.on_page_start()
        from babeldoc.format.pdf.new_parser.prepared_page import il_page_cropbox
        il_x0, il_y0, il_x1, il_y1 = il_page_cropbox(page)
        sink.on_page_crop_box(float(il_x0), float(il_y0), float(il_x1), float(il_y1))
        sink.on_page_media_box(float(x0), float(y0), float(x1), float(y1))
        sink.on_page_number(page.pageno)

        ops_base = _emit_page_to_sink(
            events=events,
            resource_bundle=resource_bundle,
            sink=sink,
            base_operations=base_operations,
            text_run_positioner=text_run_positioner,
            page=page,
        )
        sink.on_page_base_operation(ops_base)
        sink.on_page_end()

    sink.on_finish()
    return sink.create_il()
