def deploy():
    from app import create_app, db
    from flask_migrate import upgrade, migrate, init, stamp
    from app.model.user import User

    try:
        app = create_app()
        app.app_context().push()
        db.create_all()

        init()
        stamp()
        migrate()
        upgrade()

        su = User()
        su.create_admin()

        email = app.config['ADMIN_MAIL']
        passwd = app.config['ADMIN_PASSWORD']

        print(f"Inisialisasi selesai. Silakan mulai aplikasi dengan mengetik flask run lalu login sebagai ADMIN pada url http://127.0.0.1:5000/login menggunakan USERNAME: {email} dan PASSWORD: {passwd}")

    except Exception as e:
        print(f"Terjadi error: {e}")

deploy()