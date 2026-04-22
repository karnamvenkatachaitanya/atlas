# Build the React frontend
FROM node:18-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Build the Python backend
FROM python:3.11-slim
WORKDIR /app

# Copy the built frontend into the final image
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

# Install backend dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend files
COPY backend/ ./backend/
COPY env/ ./env/
COPY agents/ ./agents/

# Command to run the FastAPI app via Uvicorn
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
