#!/usr/bin/env python3
"""자막 + 제목 미리보기 테스트"""

from moviepy import ColorClip, TextClip, CompositeVideoClip, ImageClip
from PIL import Image
import numpy as np

WIDTH = 1080
HEIGHT = 1920

# 배경 (검정)
bg = ColorClip(size=(WIDTH, HEIGHT), color=(15, 15, 20))

# 샘플 이미지 영역 표시 (회색 박스로 이미지 위치 표시)
# 2.4배 확대 크롭이므로 중앙에 작은 영역만 보임
img_area = ColorClip(size=(int(WIDTH * 0.8), int(HEIGHT * 0.5)),
                     color=(60, 60, 70))
img_area = img_area.with_position(("center", "center"))

# 제목 (상단 크롭 영역)
title = TextClip(
    text="삼 년 사귄 여친의 충격 비밀",
    font_size=52,
    color="white",
    font="/System/Library/Fonts/AppleSDGothicNeo.ttc",
    method="caption",
    size=(WIDTH - 80, None),
    text_align="center",
    stroke_color="black",
    stroke_width=2,
)
title = title.with_position(("center", 70))

# 자막 (하단)
subtitle_text = "근데 알고 보니까요"
txt_clip = TextClip(
    text=subtitle_text,
    font_size=72,
    color="white",
    font="/System/Library/Fonts/AppleSDGothicNeo.ttc",
    method="caption",
    size=(WIDTH - 160, None),
    text_align="center",
    stroke_color="black",
    stroke_width=3,
)

# 자막 배경 박스
txt_w, txt_h = txt_clip.size
padding_x = 40
padding_y = 30
subtitle_bg = ColorClip(
    size=(txt_w + padding_x * 2, txt_h + padding_y * 2),
    color=(0, 0, 0),
)
subtitle_bg = subtitle_bg.with_position(("center", HEIGHT * 0.72))
txt_clip = txt_clip.with_position(("center", HEIGHT * 0.72 + padding_y))

# 안내 텍스트
guide1 = TextClip(
    text="↑ 상단: 크롭 영역 (제목)",
    font_size=30,
    color="yellow",
    font="/System/Library/Fonts/AppleSDGothicNeo.ttc",
)
guide1 = guide1.with_position(("center", 180))

guide2 = TextClip(
    text="[이미지 영역]",
    font_size=40,
    color="gray",
    font="/System/Library/Fonts/AppleSDGothicNeo.ttc",
)
guide2 = guide2.with_position(("center", HEIGHT // 2))

guide3 = TextClip(
    text="↓ 하단: 자막 영역",
    font_size=30,
    color="yellow",
    font="/System/Library/Fonts/AppleSDGothicNeo.ttc",
)
guide3 = guide3.with_position(("center", HEIGHT * 0.72 - 80))

# 합성
final = CompositeVideoClip(
    [bg, img_area, title, guide1, guide2, guide3, subtitle_bg, txt_clip],
    size=(WIDTH, HEIGHT))

# 프레임 추출해서 이미지로 저장
frame = final.get_frame(0)
img = Image.fromarray(frame)
img.save("/Users/hyungseok2/project/shorts-automation/preview_layout.png")

print("✅ 미리보기 저장됨: preview_layout.png")
print(f"   해상도: {WIDTH}x{HEIGHT}")
