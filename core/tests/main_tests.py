from pathlib import Path

from core.fastinfer import Prompt, GenParams
from core.fastinfer.config_load import load_experiment_config
from core.fastinfer.engines.fake import FakeEngineV1
from core.fastinfer.results import ExperimentWriter
from tests.test_scenario_fake import test_fake_scenario


def test_all():
    test_config_load()

    test_result_write()

    test_fake_engine()

    test_fake_scenario()





TEST_CONFIG_FILE = "../resources/test_scenario_fake.yaml"
def test_config_load():

    try:
        test_conf = load_experiment_config(TEST_CONFIG_FILE)
    except Exception as e:
        print(e)

    print("config Test passed")



def test_result_write():
    writer = ExperimentWriter(load_experiment_config(TEST_CONFIG_FILE))

    writer.start()

    writer.write_log("test log")

    print("writer test passed")


def test_fake_engine():
    prompt = Prompt(id="test/qa/1", task="qa", turns=["Привет"])
    gen = GenParams(max_tokens=16)

    with FakeEngineV1(seed=123) as engine:
        first = engine.generate(prompt, gen)
        second = engine.generate(prompt, gen)

    assert len(first) == 1
    assert first[0].n_gen == len(first[0].token_ids) > 0
    assert first[0].token_ids == second[0].token_ids
















































































