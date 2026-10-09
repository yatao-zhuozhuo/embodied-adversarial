from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api")
HTML = (Path(__file__).resolve().parents[1] / "SFT_data/fronted/static/two_tool.html").read_text()


@pytest.fixture
def viewer():
    with playwright.sync_playwright() as runtime:
        executables = sorted(Path.home().glob(".cache/ms-playwright/chromium-*/chrome-linux64/chrome"))
        options = {"executable_path": str(executables[-1])} if executables else {}
        browser = runtime.chromium.launch(headless=True, args=["--no-sandbox"], **options)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        state = {"reverse": False, "frame_requests": []}
        rows = [{"bench": "metaworld", "task_id": task, "attempt": "attempt-0000",
                 "turns": 20, "action_source": "test"} for task in ("first", "second")]

        def respond(route):
            path = route.request.url.split("/proxy/8770", 1)[1]
            if path.startswith("/api/episodes"):
                items = list(reversed(rows)) if state["reverse"] else rows
                route.fulfill(json={"total": 2, "items": items})
            elif path.startswith("/api/episode/"):
                route.fulfill(json={"bench": "metaworld", "task": "Move", "success": True,
                                    "initial_frames": ["frames/0000-camera.png"],
                                    "turns": [{"frames": [f"frames/{index:04d}-camera.png"],
                                               "tool": "step_env", "model_message": {"content": "向目标移动。"}}
                                              for index in range(1, 21)]})
            elif path.startswith("/api/frame/"):
                state["frame_requests"].append(path)
                if "0012-camera" in path:
                    route.fulfill(status=404)
                else:
                    color = "red" if "/first/" in path else "blue"
                    route.fulfill(content_type="image/svg+xml", body=f'<svg xmlns="http://www.w3.org/2000/svg" width="128" height="128"><rect width="128" height="128" fill="{color}"/></svg>')
            else:
                route.fulfill(content_type="text/html", body=HTML)

        page.route("**/*", respond)
        page.goto("http://viewer.test/proxy/8770/")
        page.locator(".item").first.click()
        page.wait_for_function("document.getElementById('position').textContent.startsWith('0 /')")
        yield page, state
        browser.close()


def test_playback_keeps_canvas_and_layout_stable(viewer):
    page, state = viewer
    assert page.evaluate("getComputedStyle(document.body).backgroundColor") == "rgb(20, 23, 28)"
    page.evaluate("window.originalCanvas = document.querySelector('canvas')")
    bounds = page.locator("canvas").bounding_box()
    page.locator("#speed").select_option("250")
    page.locator("#play").click()
    page.wait_for_function("Number(document.getElementById('slider').value) >= 3")
    page.locator("#play").click()
    assert page.evaluate("window.originalCanvas === document.querySelector('canvas')")
    assert page.locator("canvas").bounding_box() == bounds
    assert page.evaluate("Array.from(document.querySelector('canvas').getContext('2d').getImageData(20,20,1,1).data)") == [255, 0, 0, 255]
    assert len(state["frame_requests"]) >= 4


def test_refresh_seek_race_and_failed_frame_keep_correct_picture(viewer):
    page, state = viewer
    state["reverse"] = True
    page.locator("#refresh").click()
    page.wait_for_function("document.querySelector('.item').textContent.includes('second')")
    page.evaluate("""() => {
      const decode = Image.prototype.decode;
      Image.prototype.decode = function() {
        return decode.call(this).then(() => new Promise(resolve => setTimeout(resolve, this.src.includes('0010-') ? 400 : 0)));
      };
      seek(10);
      seek(11);
    }""")
    page.wait_for_function("document.getElementById('slider').value === '11'")
    page.wait_for_timeout(450)
    assert page.locator("#slider").input_value() == "11"
    assert all("/first/" in path for path in state["frame_requests"])
    before = page.locator("canvas").screenshot()
    page.evaluate("seek(12)")
    page.wait_for_function("document.getElementById('status').textContent.includes('图片加载失败')")
    assert page.locator("#slider").input_value() == "11"
    assert page.locator("canvas").screenshot() == before


def test_pause_cancels_an_inflight_frame(viewer):
    page, _ = viewer
    page.evaluate("""() => {
      const decode = Image.prototype.decode;
      Image.prototype.decode = function() {
        return decode.call(this).then(() => new Promise(resolve => setTimeout(resolve, 300)));
      };
      seek(10);
      stop();
    }""")
    page.wait_for_timeout(450)
    assert page.locator("#slider").input_value() == "0"
    assert page.locator("#play").inner_text() == "播放"
