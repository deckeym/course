FROM golang:1.22-alpine AS builder

WORKDIR /app

# Копируем сразу весь проект (main.go, go.mod, если есть)
COPY . .

# Если go.mod нет — создаём, затем подтягиваем зависимости и формируем go.sum
RUN if [ ! -f go.mod ]; then \
      go mod init traffic/go_incidents; \
    fi && \
    go mod tidy

# Сборка бинарника
RUN go build -o go_incidents

FROM alpine:3.20
WORKDIR /app
COPY --from=builder /app/go_incidents .
ENV PORT=8080
CMD ["./go_incidents"]
