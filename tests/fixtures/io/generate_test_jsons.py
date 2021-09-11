# Copyright 2021 The Layout Parser team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import numpy as np
from layoutparser.elements import Interval, Rectangle, Quadrilateral, TextBlock, Layout

if __name__ == "__main__":

    i = Interval(1, 2, "y", canvas_height=5)
    r = Rectangle(1, 2, 3, 4)
    q = Quadrilateral(np.arange(8).reshape(4, 2), 200, 400)
    l = Layout([i, r, q], page_data={"width": 200, "height": 200})

    with open("interval.json", "w") as fp:
        json.dump(i.to_dict(), fp)
    with open("rectangle.json", "w") as fp:
        json.dump(r.to_dict(), fp)
    with open("quadrilateral.json", "w") as fp:
        json.dump(q.to_dict(), fp)
    with open("layout.json", "w") as fp:
        json.dump(l.to_dict(), fp)
    l.to_dataframe().to_csv("layout.csv", index=None)

    i2 = TextBlock(i, "")
    r2 = TextBlock(r, id=24)
    q2 = TextBlock(q, text="test", parent=45)
    l2 = Layout([i2, r2, q2])

    with open("interval_textblock.json", "w") as fp:
        json.dump(i2.to_dict(), fp)
    with open("rectangle_textblock.json", "w") as fp:
        json.dump(r2.to_dict(), fp)
    with open("quadrilateral_textblock.json", "w") as fp:
        json.dump(q2.to_dict(), fp)
    with open("layout_textblock.json", "w") as fp:
        json.dump(l2.to_dict(), fp)
    l2.to_dataframe().to_csv("layout_textblock.csv", index=None)