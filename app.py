from flask import Flask, render_template, request, redirect, url_for, session, flash
from datetime import datetime
from flask_migrate import Migrate
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import os
import requests

from models import db, PassengerData, Prediction, User, Incident
from sqlalchemy.exc import DataError
from functools import wraps

# Следующие импорты не используются, можно удалить при желании
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:4780@db/traffic_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = "supersecretkey"

GO_API_URL = os.getenv("GO_API_URL", "http://go_api:8080")

# До какого года данные считаются фактическими (остальное — прогноз)
MAX_REAL_YEAR = int(os.getenv("MAX_REAL_YEAR", "2025"))

db.init_app(app)
migrate = Migrate(app, db)


def is_admin():
    return session.get("is_admin", False)


def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get("is_admin"):
            flash("У вас нет прав для этого действия.", "auth_danger")
            return redirect(url_for("dashboard"))
        return f(*args, **kwargs)
    return wrapper


with app.app_context():
    db.create_all()

    # создаём админа, если его нет
    admin = User.query.filter_by(username="admin").first()
    if not admin:
        admin = User(username="admin", is_admin=True)
        admin.set_password("admin123")
        db.session.add(admin)
        db.session.commit()
        print("✅ Админ создан: admin/admin123")

    # синтетические данные, если таблица пустая
    if PassengerData.query.count() == 0:
        print("📊 Инициализация синтетических данных...")
        # Делаем данные сразу до 2030 года, чтобы инциденты могли ссылаться
        years = list(range(2016, 2031))
        months = list(range(1, 13))
        seasonal_factor = {
            1: 70, 2: 75, 3: 80, 4: 85, 5: 95, 6: 110,
            7: 130, 8: 125, 9: 100, 10: 90, 11: 80, 12: 95
        }
        for y in years:
            for m in months:
                base = 90
                passengers = base + seasonal_factor.get(m, 0) + np.random.normal(0, 3)
                db.session.add(
                    PassengerData(year=y, month=m, passengers=int(round(passengers)))
                )
        db.session.commit()
        print("✅ Синтетические данные добавлены.")


@app.route('/')
def index():
    return render_template('index.html', current_year=datetime.now().year)


# РЕГИСТРАЦИЯ
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        raw_username = request.form.get('username', '')
        raw_password = request.form.get('password', '')

        username = raw_username.strip()
        password = raw_password

        if not username or not password:
            flash("Логин и пароль не могут быть пустыми.", "auth_danger")
            return render_template('register.html')

        if any(ch.isspace() for ch in username) or any(ch.isspace() for ch in password):
            flash("Логин и пароль не должны содержать пробелы.", "auth_danger")
            return render_template('register.html')

        if len(username) < 3 or len(password) < 8:
            flash("Логин — минимум 3 символа, пароль — минимум 8 символов.", "auth_danger")
            return render_template('register.html')

        if User.query.filter_by(username=username).first():
            flash("Имя пользователя уже занято.", "auth_danger")
            return render_template('register.html')

        new_user = User(username=username)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        flash("Регистрация успешна! Теперь войдите.", "auth_success")
        return redirect(url_for("login"))

    return render_template('register.html')


# ЛОГИН
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            session.clear()
            session['user_id'] = user.id
            session['username'] = username
            session['is_admin'] = user.is_admin
            flash("Вы успешно вошли!", "auth_success")
            return redirect(url_for("dashboard"))
        else:
            flash("Неверное имя пользователя или пароль.", "auth_danger")

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash("Вы вышли из аккаунта.", "auth_info")
    return redirect(url_for("login"))


# ЛИЧНЫЙ КАБИНЕТ
@app.route('/dashboard')
def dashboard():
    if "user_id" not in session:
        flash("Войдите в аккаунт.", "auth_warning")
        return redirect(url_for("login"))

    user = db.session.get(User, session["user_id"])
    return render_template(
        'dashboard.html',
        username=user.username,
        is_admin=user.is_admin
    )


# РЕДАКТИРОВАНИЕ ДАННЫХ — ТОЛЬКО АДМИН
@app.route('/edit', methods=['GET', 'POST'])
@admin_required
def edit_data():
    if request.method == 'POST':
        try:
            year = int(request.form['year'])
            month = int(request.form['month'])
            passengers = int(request.form['passengers'])

            # редактируем только фактический диапазон
            if not (2016 <= year <= MAX_REAL_YEAR):
                flash(
                    f"Редактирование возможно только для фактического периода 2016–{MAX_REAL_YEAR}.",
                    "edit_danger",
                )
                return redirect(url_for('edit_data'))

            if not (1 <= month <= 12):
                flash("Месяц должен быть от 1 до 12.", "edit_danger")
                return redirect(url_for('edit_data'))

            if passengers < 0:
                flash("Количество пассажиров поездов не может быть отрицательным.", "edit_danger")
                return redirect(url_for('edit_data'))

            MAX_PASSENGERS = 2_000_000_000
            if passengers > MAX_PASSENGERS:
                flash(
                    f"Слишком большое значение пассажиров поездов. Максимум: {MAX_PASSENGERS}.",
                    "edit_danger",
                )
                return redirect(url_for('edit_data'))

            entry = PassengerData.query.filter_by(year=year, month=month).first()
            if not entry:
                flash(f"Данные за {year}-{month:02d} не найдены.", "edit_warning")
                return redirect(url_for('edit_data'))

            entry.passengers = passengers

            try:
                db.session.commit()
            except DataError:
                db.session.rollback()
                flash("Ошибка: число превышает допустимый диапазон.", "edit_danger")
                return redirect(url_for('edit_data'))

            flash(f"Данные за {year}-{month:02d} обновлены.", "edit_success")
            return redirect(url_for('dashboard'))

        except ValueError:
            flash("Проверьте корректность введённых значений.", "edit_danger")
            return redirect(url_for('edit_data'))

    # в превью показываем только фактические годы
    records = (
        PassengerData.query
        .filter(PassengerData.year <= MAX_REAL_YEAR)
        .order_by(PassengerData.year, PassengerData.month)
        .all()
    )
    data_map = {f"{r.year}-{r.month:02d}": r.passengers for r in records}
    return render_template(
        'edit_data.html',
        records=records,
        data_map=data_map,
        max_real_year=MAX_REAL_YEAR,
    )


# ПРОГНОЗ — ДОСТУПЕН ВСЕМ ПОЛЬЗОВАТЕЛЯМ
@app.route('/predict', methods=['GET', 'POST'])
def predict():

    prediction = None
    plot_url = None

    if request.method == 'POST':
        try:
            year = int(request.form['year'])
            month = int(request.form['month'])

            # --- ВАЛИДАЦИЯ ДАТЫ ПРОГНОЗА ---

            # 1. Месяц 1–12
            if not (1 <= month <= 12):
                flash("Месяц должен быть от 1 до 12.", "danger")
                return redirect(url_for("predict"))

            # 2. Верхняя граница — не позже декабря 2030 года
            if year > 2030 or (year == 2030 and month > 12):
                flash("Прогноз можно строить не позже декабря 2030 года.", "danger")
                return redirect(url_for("predict"))

            # 3. Минимум — следующий месяц относительно текущей даты
            today = datetime.now()
            cur_year = today.year
            cur_month = today.month

            next_year = cur_year
            next_month = cur_month + 1
            if next_month > 12:
                next_month = 1
                next_year += 1

            # если запрошенный период раньше первого допустимого — запрещаем
            if year < next_year or (year == next_year and month < next_month):
                flash(
                    f"Прогноз можно строить только для будущих периодов, "
                    f"начиная с {next_month:02d}.{next_year}.",
                    "danger",
                )
                return redirect(url_for("predict"))

            # --- /ВАЛИДАЦИЯ ДАТЫ ПРОГНОЗА ---

            # Берём только фактические данные до MAX_REAL_YEAR
            data = (
                PassengerData.query
                .filter(PassengerData.year <= MAX_REAL_YEAR)
                .all()
            )
            # Инциденты могут быть и после MAX_REAL_YEAR — они влияют на будущий прогноз
            incidents = Incident.query.all()

            rows = [{'year': d.year, 'month': d.month, 'passengers': d.passengers} for d in data]
            df = pd.DataFrame(rows)
            df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))

            incident_effects = pd.Series(
                1.0,
                index=pd.date_range(start='2016-01-01', end='2030-12-01', freq='MS')
            )

            for inc in incidents:
                start = pd.to_datetime(f"{inc.year}-{inc.month:02d}-01")
                duration = inc.duration or 1
                for i in range(duration):
                    month_inc = start + pd.DateOffset(months=i)
                    if month_inc in incident_effects.index:
                        decay = 1 + inc.impact * (1 - i / duration)
                        incident_effects[month_inc] *= decay

            df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
            df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)

            pred_date = pd.to_datetime(f"{year}-{month:02d}-01")

            model = LinearRegression()
            X = df[['year', 'sin_month', 'cos_month']]
            y = df['passengers']
            model.fit(X, y)

            start_date = pred_date - pd.DateOffset(years=2)
            full_range = pd.date_range(start=start_date, end=pred_date, freq='MS')

            df_full = pd.DataFrame({'date': full_range})
            df_full['year'] = df_full['date'].dt.year
            df_full['month'] = df_full['date'].dt.month
            df_full['sin_month'] = np.sin(2 * np.pi * df_full['month'] / 12)
            df_full['cos_month'] = np.cos(2 * np.pi * df_full['month'] / 12)

            df_full = pd.merge(df_full, df[['date', 'passengers']], on='date', how='left')

            missing = df_full['passengers'].isna()
            if missing.any():
                X_missing = df_full.loc[missing, ['year', 'sin_month', 'cos_month']]
                predicted_missing = model.predict(X_missing)
                df_full.loc[missing, 'passengers'] = predicted_missing

            df_full['adjusted'] = df_full['passengers']

            for date in df_full['date']:
                if date in incident_effects.index:
                    df_full.loc[df_full['date'] == date, 'adjusted'] *= incident_effects[date]

            prediction = df_full.loc[df_full['date'] == pred_date, 'adjusted'].values[0]

            # 🔗 сохраняем прогноз, привязывая к пользователю (user_id)
            user_id = session.get("user_id")
            pred = Prediction(
                year=year,
                month=month,
                predicted_passengers=int(prediction),
                user_id=user_id,
            )
            db.session.add(pred)
            db.session.commit()

            plt.figure(figsize=(10, 4))
            is_real = df_full['date'].isin(df['date'])
            is_pred = ~is_real

            plt.plot(
                df_full.loc[is_real, 'date'],
                df_full.loc[is_real, 'adjusted'],
                marker='o',
                label='Факт'
            )

            plt.plot(
                df_full.loc[is_pred, 'date'],
                df_full.loc[is_pred, 'adjusted'],
                marker='o',
                linestyle='dashed',
                label='Прогноз'
            )

            plt.scatter(
                [pred_date],
                [prediction],
                color='red',
                label=f'Прогноз ({prediction:.0f})',
                zorder=5
            )
            plt.annotate(
                f'{prediction:.0f}',
                xy=(pred_date, prediction),
                xytext=(5, 5),
                textcoords='offset points',
                color='red'
            )

            plt.xticks(df_full['date'], df_full['date'].dt.strftime('%Y-%m'), rotation=45)
            plt.title('Пассажиропоток поездов с прогнозом')
            plt.ylabel('Количество пассажиров поездов')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            plt.close()

        except Exception as e:
            flash(f"Ошибка вычисления прогноза: {str(e)}", "danger")
            return redirect(url_for("predict"))

    return render_template(
        'predict.html',
        prediction=prediction,
        plot_url=plot_url,
        is_admin=is_admin(),
        max_real_year=MAX_REAL_YEAR,
    )


# СТАТИСТИКА — ДОСТУПНА ВСЕМ ПОЛЬЗОВАТЕЛЯМ
@app.route('/statistics', methods=['GET', 'POST'])
def statistics():

    selected_year = None
    rows = []

    if request.method == 'POST':
        year_raw = request.form.get('year', '').strip()

        try:
            year = int(year_raw)
        except (TypeError, ValueError):
            flash("Введите корректный год.", "danger")
            return redirect(url_for('statistics'))

        # по условию – только 2016–2024
        if not (2016 <= year <= 2024):
            flash("Год должен быть в диапазоне 2016–2024.", "danger")
            return redirect(url_for('statistics'))

        selected_year = year

        # данные пассажиропотока за выбранный год
        data = (
            PassengerData.query
            .filter_by(year=year)
            .order_by(PassengerData.month)
            .all()
        )
        data_by_month = {d.month: d for d in data}

        # все инциденты (будем отмечать месяцы, на которые они влияют)
        incidents = Incident.query.all()

        # месяц -> список текстовых описаний инцидентов
        month_incidents = {m: [] for m in range(1, 13)}

        for inc in incidents:
            start_month = inc.month
            start_year = inc.year
            duration = inc.duration or 1

            for i in range(duration):
                m = start_month + i
                y = start_year

                # корректируем год/месяц, если вышли за пределы года
                while m > 12:
                    m -= 12
                    y += 1

                if y == year and 1 <= m <= 12:
                    desc = inc.description or "Инцидент"
                    text = f"{desc} (влияние {inc.impact:.2f}, {duration} мес.)"
                    month_incidents[m].append(text)

        month_names = {
            1: "Январь", 2: "Февраль", 3: "Март", 4: "Апрель",
            5: "Май", 6: "Июнь", 7: "Июль", 8: "Август",
            9: "Сентябрь", 10: "Октябрь", 11: "Ноябрь", 12: "Декабрь",
        }

        for m in range(1, 13):
            record = data_by_month.get(m)
            total = record.passengers if record else 0

            # временное разбиение на Пригородное / Дальнее следование
            suburban = int(total * 0.7)
            long_distance = total - suburban

            inc_list = month_incidents[m]

            rows.append({
                "month": m,
                "month_name": month_names[m],
                "suburban": suburban,
                "long_distance": long_distance,
                "total": total,
                "incidents": inc_list,
                "has_incident": bool(inc_list),
            })

    return render_template(
        'statistics.html',
        selected_year=selected_year,
        rows=rows,
        is_admin=is_admin(),
    )


# ИНЦИДЕНТЫ — ТОЛЬКО АДМИН
@app.route('/incidents', methods=['GET', 'POST'])
@admin_required
def incidents():
    if request.method == 'POST':
        payload = {
            "year":        request.form.get("year"),
            "month":       request.form.get("month"),
            "duration":    request.form.get("duration"),
            "impact":      request.form.get("impact"),
            "description": request.form.get("description") or "",
        }

        try:
            resp = requests.post(f"{GO_API_URL}/api/incidents", json=payload, timeout=5)
            data = resp.json()
        except Exception as e:
            flash(f"Ошибка обращения к Go-сервису: {e}", "inc_danger")
            return redirect(url_for("incidents"))

        if resp.status_code != 201:
            flash(data.get("error", "Ошибка Go-сервиса."), "inc_danger")
            return redirect(url_for("incidents"))

        flash("Инцидент успешно добавлен.", "inc_success")
        return redirect(url_for("incidents"))

    incidents_list = []
    try:
        resp = requests.get(f"{GO_API_URL}/api/incidents", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            incidents_list = data.get("incidents", [])
        else:
            flash("Не удалось получить список инцидентов.", "inc_danger")
    except Exception as e:
        flash(f"Ошибка обращения к Go-сервису: {e}", "inc_danger")

    return render_template("incidents.html", incidents=incidents_list)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
