runserver:
	docker-compose up

migrations:
	docker-compose run app python manage.py makemigrations

migrate:
	docker-compose run app python manage.py migrate

createsuperuser:
	docker-compose run app python manage.py createsuperuser

createuser:
	docker-compose run app python manage.py createuser

static:
	docker-compose run app python manage.py collectstatic --no-input

translate:
	docker-compose run app python manage.py compilemessages

shell:
	docker-compose run app python manage.py shell

test:
	docker-compose run app pytest

docker-production:
	docker build . -t production
	docker run -it --mount type=bind,src="$(shell pwd)",dst=/app production:latest python manage.py compilemessages
	docker run -it --mount type=bind,src="$(shell pwd)",dst=/app production:latest python manage.py collectstatic --no-input
	docker run -p 8000:8000 production

down:
	docker-compose down

teardown:
	docker-compose down --remove-orphans --volumes
	docker-compose kill
	docker-compose rm -f -v