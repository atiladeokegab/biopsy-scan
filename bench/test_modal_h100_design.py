import json

import pytest

from bench.modal_h100_design import ARCHIVE, archived_reference, validate_records


def test_all_historical_gpu_studies_match_fused_reference():
    receipt = json.loads(ARCHIVE.read_text())
    reference = archived_reference(receipt)
    gpu = receipt['arms']['gpu']['studies']
    parity = validate_records(gpu, reference)
    assert parity['studies_compared'] == 3000 and parity['max_abs_maxT_delta'] < 1e-9

    with pytest.raises(ValueError, match='missing, duplicated'):
        validate_records(gpu[:-1] + gpu[:1], reference)
    with pytest.raises(ValueError, match='feature hash'):
        validate_records([gpu[0] | {'x_sha': 'different'}, *gpu[1:]], reference)
    with pytest.raises(ValueError, match='absolute 1e-9'):
        validate_records([gpu[0] | {'maxT': gpu[0]['maxT'] + 1e-7}, *gpu[1:]], reference)
