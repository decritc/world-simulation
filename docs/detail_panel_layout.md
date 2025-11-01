"""Layout verification for detail panel.

Panel Layout (from bottom to top):
===================================

Bottom of panel (panel_y = 20):
  - Stamina bar: y = 20 + 18 + 15 = 53
    - Label "Stamina:" at y = 53 + 15 + 18 = 86
  - Hunger bar: y = 53 + 50 + 15 = 118
    - Label "Hunger:" at y = 118 + 15 + 18 = 151
  - Health bar: y = 118 + 50 + 15 = 183
    - Label "Health:" at y = 183 + 15 + 18 = 216

Bar area height = 3 * (15 + 18) + 2 * (50 - 18) = 99 + 64 = 163px
Health label top = 216px

Gap above health label = 25px
NN info bottom = 216 + 25 = 241px
NN info area height = 48px (4 lines * 12px)
NN visualization bottom = 241 - 48 = 193px
NN visualization height = 180px
NN visualization top = 193 + 180 = 373px
NN title is at top of visualization area

Total bottom area = 163 + 25 + 48 + 180 + 30 = 446px

Text area starts at: panel_y + panel_height - 30
Text area ends at: panel_y + total_bottom_area + 50 = 20 + 446 + 50 = 516px

For 720px height window:
- panel_height = 720 - 40 = 680px
- text_start_y = 20 + 680 - 30 = 670px
- text_end_y = 516px
- scrollable_height = 670 - 516 = 154px

All elements should fit within window bounds.
"""
