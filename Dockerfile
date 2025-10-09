# Используем официальный Python-образ
FROM python:3.9

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы проекта в контейнер
COPY . .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Открываем порт Flask
EXPOSE 5000

# Запускаем Flask-приложение
CMD ["python", "app.py"]