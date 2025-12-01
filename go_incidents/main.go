package main

import (
	"database/sql"
	"encoding/json"
	"errors"
	"log"
	"net/http"
	"os"
	"strconv"
	"time"

	_ "github.com/lib/pq"
)

type Incident struct {
	ID          int64   `json:"id"`
	Year        int     `json:"year"`
	Month       int     `json:"month"`
	Duration    int     `json:"duration"`
	Impact      float64 `json:"impact"`
	Description string  `json:"description"`
}

var db *sql.DB

func main() {
	dsn := os.Getenv("DATABASE_URL")
	if dsn == "" {
		// тот же логин/пароль, что и у Flask
		dsn = "postgres://postgres:4780@db/traffic_db?sslmode=disable"
	}

	var err error
	db, err = sql.Open("postgres", dsn)
	if err != nil {
		log.Fatalf("open db: %v", err)
	}
	db.SetMaxOpenConns(5)
	db.SetMaxIdleConns(5)
	db.SetConnMaxLifetime(5 * time.Minute)

	if err = db.Ping(); err != nil {
		log.Fatalf("ping db: %v", err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", healthHandler)
	mux.HandleFunc("/api/incidents", incidentsHandler)

	addr := ":8080"
	if port := os.Getenv("PORT"); port != "" {
		addr = ":" + port
	}
	log.Printf("Go incidents service listening on %s", addr)

	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatal(err)
	}
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("ok"))
}

func incidentsHandler(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		listIncidents(w, r)
	case http.MethodPost:
		createIncident(w, r)
	default:
		w.WriteHeader(http.StatusMethodNotAllowed)
	}
}

func listIncidents(w http.ResponseWriter, r *http.Request) {
	rows, err := db.Query(`
        SELECT id, year, month, duration, impact, COALESCE(description, '')
        FROM incident
        ORDER BY year, month, id
    `)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "db query error: "+err.Error())
		return
	}
	defer rows.Close()

	var result []Incident
	for rows.Next() {
		var inc Incident
		if err := rows.Scan(&inc.ID, &inc.Year, &inc.Month, &inc.Duration, &inc.Impact, &inc.Description); err != nil {
			writeError(w, http.StatusInternalServerError, "scan error: "+err.Error())
			return
		}
		result = append(result, inc)
	}

	writeJSON(w, http.StatusOK, map[string]any{"incidents": result})
}

func createIncident(w http.ResponseWriter, r *http.Request) {
	var payload struct {
		Year        string `json:"year"`
		Month       string `json:"month"`
		Duration    string `json:"duration"`
		Impact      string `json:"impact"`
		Description string `json:"description"`
	}

	if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
		writeError(w, http.StatusBadRequest, "invalid json: "+err.Error())
		return
	}

	year, err := strconv.Atoi(payload.Year)
	if err != nil {
		writeError(w, http.StatusBadRequest, "год должен быть числом")
		return
	}
	month, err := strconv.Atoi(payload.Month)
	if err != nil {
		writeError(w, http.StatusBadRequest, "месяц должен быть числом")
		return
	}
	duration, err := strconv.Atoi(payload.Duration)
	if err != nil {
		writeError(w, http.StatusBadRequest, "длительность должна быть числом")
		return
	}
	impact, err := strconv.ParseFloat(payload.Impact, 64)
	if err != nil {
		writeError(w, http.StatusBadRequest, "impact должен быть числом")
		return
	}

	inc := Incident{
		Year:        year,
		Month:       month,
		Duration:    duration,
		Impact:      impact,
		Description: payload.Description,
	}

	if err := validateIncident(&inc); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	err = db.QueryRow(`
        INSERT INTO incident (year, month, duration, impact, description)
        VALUES ($1, $2, $3, $4, NULLIF($5, ''))
        RETURNING id
    `, inc.Year, inc.Month, inc.Duration, inc.Impact, inc.Description).Scan(&inc.ID)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "db insert error: "+err.Error())
		return
	}

	writeJSON(w, http.StatusCreated, inc)
}

func validateIncident(inc *Incident) error {
	if inc.Year < 2026 || inc.Year > 2030 {
		return errors.New("год должен быть от 2026 до 2030")
	}
	if inc.Month < 1 || inc.Month > 12 {
		return errors.New("месяц должен быть от 1 до 12")
	}
	if len(inc.Description) > 255 {
		return errors.New("описание слишком длинное (макс. 255 символов)")
	}
	if inc.Duration <= 0 {
		return errors.New("длительность должна быть положительной")
	}
	return nil
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		log.Printf("write json error: %v", err)
	}
}
