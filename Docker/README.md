# Imagen Docker - Penguins API

Construir la imagen (desde la **raíz del proyecto**):

```bash
docker build -f Docker/Dockerfile -t penguins-api .
```

Ejecutar el contenedor (API en puerto **8989**):

```bash
docker run -p 8989:8989 penguins-api
```

- API: http://localhost:8989
- Docs: http://localhost:8989/docs
- POST /rf y POST /lr para predicciones
