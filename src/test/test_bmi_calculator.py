import pytest
from practice.bmi_calculator import calculate_bmi

def test_calculate_bmi():
    assert float(calculate_bmi(23, 43)) == 8.75
    #exception에 대한 테스트
    with pytest.raises(TypeError):
        calculate_bmi('34', '34')