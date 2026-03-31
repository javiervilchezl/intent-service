# Servicio de Deteccion de Intencion

Microservicio FastAPI para deteccion de intencion y extraccion de entidades desde texto libre.

## Responsabilidad

- inferir la intencion principal de un mensaje
- extraer entidades relevantes
- responder en formato JSON consistente para automatizacion y routing

## Endpoint

- `POST /detect-intent`

Entrada:

```json
{
  "text": "Quiero reservar un vuelo a Madrid"
}
```

Salida:

```json
{
  "intent": "reservar_vuelo",
  "entities": {
    "destination": "Madrid"
  }
}
```

## Variables de entorno

- `PROVIDER` (`openai` o `groq`)
- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `GROQ_API_KEY`
- `GROQ_MODEL`

## Ejecucion local

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8003
```

## Docker

```bash
docker compose up --build
```

## Pruebas

```bash
pip install -r requirements-dev.txt
pytest
```

Cobertura configurada al 100%.

