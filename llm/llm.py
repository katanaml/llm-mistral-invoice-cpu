from langchain.llms import CTransformers
import box
import yaml


# Import config vars
with open('config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


def setup_llm():
    llm = CTransformers(model=cfg.MODEL_BIN_PATH,
                        model_type=cfg.MODEL_TYPE,
                        max_new_tokens=cfg.MAX_NEW_TOKENS,
                        temperature=cfg.TEMPERATURE
    )

    return llm
