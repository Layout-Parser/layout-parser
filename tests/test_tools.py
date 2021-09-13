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

from layoutparser import load_pdf
from layoutparser.tools import (
    generalized_connected_component_analysis_1d,
    simple_line_detection,
    group_textblocks_based_on_category,
)

def test_generalized_connected_component_analysis_1d():
    
    A = [1, 2, 3]
    
    results = generalized_connected_component_analysis_1d(
        A,
        scoring_func=lambda x,y: abs(x-y)<=1
    )
    assert len(results) == 1
    
    A = [1, 2, 3, 5, 6, 7]
    results = generalized_connected_component_analysis_1d(
        A,
        scoring_func=lambda x,y: abs(x-y)<=1
    )
    assert len(results) == 2
    
    A = [1, 2, 3, 5, 6, 7]
    results = generalized_connected_component_analysis_1d(
        A,
        scoring_func=lambda x,y: abs(x-y)<=2
    )
    assert len(results) == 1
    
    A = [1, 2, 3, 5, 6, 7]
    results = generalized_connected_component_analysis_1d(
        A,
        scoring_func=lambda x,y: abs(x-y)<=1,
        aggregation_func=max
    )
    assert results == [3, 7]
    
def test_simple_line_detection():
    
    page_layout = load_pdf("tests/fixtures/io/example.pdf")[0]
    
    pdf_lines = simple_line_detection(page_layout)
    
    assert len(pdf_lines) == 15
    
def test_group_textblocks_based_on_category():
    
    page_layout = load_pdf("tests/fixtures/io/example.pdf")[0]
    
    pdf_blocks = group_textblocks_based_on_category(page_layout)
    
    assert len(pdf_blocks) == 3