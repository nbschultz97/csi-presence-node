import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import yaml
from csi_node import pipeline


def test_offline():
    cfg = yaml.safe_load(open("csi_node/config.yaml"))
    df = pipeline.run_offline("tests/data/sample.log", cfg)
    print(df)
    assert not df.empty


if __name__ == "__main__":
    test_offline()
