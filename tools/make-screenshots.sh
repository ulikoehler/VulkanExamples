#!/bin/bash
# Regenerate screenshots/ — runs each example, then its check.py
# (which verifies pixels/logs), then converts any produced image
# artifacts to PNG in screenshots/. Windowed-only examples get
# captured via Xvfb + ImageMagick `import`.
#
# Needs: convert, montage, import (imagemagick), xvfb-run.
set -u
cd "$(dirname "$0")/.."
mkdir -p screenshots

# Never run p40's decode path on a dGPU — the Vulkan Video
# decode-submit can crash drivers; it's verified separately via
# VK_ICD_FILENAMES on a GPU without a video queue (clean SKIP).
SKIP_RUN="p40_video p18-validation"

for d in p*/; do
    d=${d%/}
    [ -f "$d/main.cpp" ] || continue
    case " $SKIP_RUN " in *" $d "*) echo "== $d: skipped"; continue;; esac
    rm -f "$d"/*.ppm "$d"/*.png 2>/dev/null

    if grep -q subprocess "$d/check.py" 2>/dev/null; then
        # check.py runs the app itself (handles xvfb where needed)
        (cd "$d" && timeout 180 python3 check.py >/dev/null 2>&1) \
            || echo "== $d: check failed"
    else
        # two-step: run the app, then verify its artifacts
        if grep -q '"--headless"' "$d/main.cpp"; then
            (cd "$d" && timeout 120 ./app --headless out.ppm \
                >/dev/null 2>&1)
        else
            (cd "$d" && timeout 120 ./app >/dev/null 2>&1)
        fi
        (cd "$d" && timeout 60 python3 check.py >/dev/null 2>&1) \
            || echo "== $d: check failed"
    fi

    imgs=()
    while IFS= read -r f; do imgs+=("$f"); done \
        < <(ls "$d"/*.ppm "$d"/*.png 2>/dev/null)
    if [ "${#imgs[@]}" -gt 0 ]; then
        if [ "${#imgs[@]}" -gt 1 ]; then
            montage "${imgs[@]}" -tile x1 -geometry +2+2 \
                -background '#202020' "screenshots/$d.png"
        else
            convert "${imgs[0]}" "screenshots/$d.png"
        fi
        echo "== $d: ${#imgs[@]} image(s)"
    fi
done

# windowed-only examples: capture the Xvfb root while running
for d in p12-resize p36-skeleton p46_hdr; do
    [ -f "$d/app" ] || continue
    args=""; [ "$d" = "p46_hdr" ] && args="--hold"
    rm -f /tmp/wcap-*.png
    timeout 90 xvfb-run -a -s "-screen 0 640x480x24" bash -c "
        cd $d
        ./app $args &
        app=\$!
        for i in \$(seq 1 15); do
            sleep 0.4
            import -window root /tmp/wcap-\$i.png 2>/dev/null
            kill -0 \$app 2>/dev/null || break
        done
        wait \$app" >/dev/null 2>&1
    for f in /tmp/wcap-*.png; do
        [ -f "$f" ] || continue
        sd=$(convert "$f" -format "%[standard-deviation]" info: 2>/dev/null)
        if python3 -c "exit(0 if float('$sd' or 0) > 1000 else 1)"; then
            convert "$f" "screenshots/$d.png"
            echo "== $d: window captured"
            break
        fi
    done
done
echo "done: $(ls screenshots/*.png 2>/dev/null | wc -l) screenshots"
