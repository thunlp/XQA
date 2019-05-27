
from os.path import join, expanduser, dirname

"""
Global config options
"""

DATA_DIR = "/data3/private/liujiahua/new/data"
TRANS_DATA_DIR = "/data1/private/liujiahua"

NEW_EN = join(DATA_DIR, "en")
NEW_EN_TRANS_DE = join(TRANS_DATA_DIR, "_wiki_data_en/translate_de")
NEW_EN_TRANS_ZH = join(TRANS_DATA_DIR, "_wiki_data_en/translate_zh")
NEW_FR_TRANS = join(TRANS_DATA_DIR, "wiki_data_fr/translate_en")
NEW_DE_TRANS = join(TRANS_DATA_DIR, "wiki_data_de/translate_en")
NEW_RU_TRANS = join(TRANS_DATA_DIR, "wiki_data_ru/translate_en")
NEW_PT_TRANS = join(TRANS_DATA_DIR, "wiki_data_pt/translate_en")
NEW_ZH_TRANS = join(TRANS_DATA_DIR, "wiki_data_zh/translate_en")
NEW_PL_TRANS = join(TRANS_DATA_DIR, "wiki_data_pl/translate_en")
NEW_UK_TRANS = join(TRANS_DATA_DIR, "wiki_data_uk/translate_en")
NEW_TA_TRANS = join(TRANS_DATA_DIR, "wiki_data_ta/translate_en")
NEW_FR_ORI = join(DATA_DIR, "fr")
NEW_DE_ORI = join(DATA_DIR, "de")
NEW_RU_ORI = join(DATA_DIR, "ru")
NEW_PT_ORI = join(DATA_DIR, "pt")
NEW_ZH_ORI = join(DATA_DIR, "zh")
NEW_PL_ORI = join(DATA_DIR, "pl")
NEW_UK_ORI = join(DATA_DIR, "uk")
NEW_TA_ORI = join(DATA_DIR, "ta")

CORPUS_NAME_TO_PATH = {
  "en": NEW_EN,
  "en_trans_de": NEW_EN_TRANS_DE,
  "en_trans_zh": NEW_EN_TRANS_ZH,
  "fr_trans_en": NEW_FR_TRANS,
  "de_trans_en": NEW_DE_TRANS,
  "ru_trans_en": NEW_RU_TRANS,
  "pt_trans_en": NEW_PT_TRANS,
  "zh_trans_en": NEW_ZH_TRANS,
  "pl_trans_en": NEW_PL_TRANS,
  "uk_trans_en": NEW_UK_TRANS,
  "ta_trans_en": NEW_TA_TRANS,
  "fr": NEW_FR_ORI,
  "de": NEW_DE_ORI,
  "ru": NEW_RU_ORI,
  "pt": NEW_PT_ORI,
  "zh": NEW_ZH_ORI,
  "pl": NEW_PL_ORI,
  "uk": NEW_UK_ORI,
  "ta": NEW_TA_ORI,
}

CORPUS_DIR = join(dirname(__file__), "data")

