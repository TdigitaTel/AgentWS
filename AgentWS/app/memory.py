_MEMORY = {}

MAX_MESSAGES = 10  # límite de historial (ajustable)


def get_context(session_id):
    """
    Devuelve el contexto completo de la sesión.
    """
    return _MEMORY.get(session_id, {})


def set_context(session_id, data):
    """
    Guarda SOLO contexto estructurado (no mensajes).
    """
    session = _MEMORY.setdefault(session_id, {})

    for k, v in data.items():
        if k != "messages":   # 🔒 blindaje clave
            session[k] = v


def add_message(session_id, role, content):
    """
    Guarda historial conversacional limitado.
    """
    session = _MEMORY.setdefault(session_id, {})
    messages = session.setdefault("messages", [])

    messages.append({
        "role": role,
        "content": content
    })

    # 🔁 Mantener solo los últimos N mensajes
    if len(messages) > MAX_MESSAGES:
        session["messages"] = messages[-MAX_MESSAGES:]