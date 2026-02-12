import yaml
from pathlib import Path
from app.logger import get_logger

logger = get_logger("DOMAIN_CONFIG")

CONFIG_PATH = Path("config/domain.yaml")

try:
    logger.info(f"Cargando configuración de dominios desde: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    DOMAINS_CONFIG = config.get("domains", [])

    if not DOMAINS_CONFIG:
        logger.warning("No se encontraron dominios en el archivo YAML")

    DOMAINS_VALIDOS = {d["nombre"] for d in DOMAINS_CONFIG}
    HANDLERS = {d["nombre"]: d["handler"] for d in DOMAINS_CONFIG}
    LIMITS_DOMAIN = {d["nombre"]: d.get("max_results")for d in DOMAINS_CONFIG if "max_results" in d}

    logger.info(f"Dominios válidos cargados: {DOMAINS_VALIDOS}")
    logger.info(f"Handlers registrados: {HANDLERS}")

except FileNotFoundError:
    logger.exception(f"No se encontró el archivo de configuración: {CONFIG_PATH}")
    DOMAINS_CONFIG = []
    DOMAINS_VALIDOS = set()
    HANDLERS = {}

except Exception as e:
    logger.exception(f"Error cargando configuración de dominios: {e}")
    DOMAINS_CONFIG = []
    DOMAINS_VALIDOS = set()
    HANDLERS = {}