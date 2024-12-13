VERSION=v1
DOCKERUSER=nidhisankhe1

build:
	docker build -f Dockerfile -t quantaas-backend-service .

push:
	docker tag quantaas-backend-service $(DOCKERUSER)/quantaas-backend-service:$(VERSION)
	docker push $(DOCKERUSER)/quantaas-backend-service:$(VERSION)
	docker tag quantaas-backend-service $(DOCKERUSER)/quantaas-backend-service:latest
	docker push $(DOCKERUSER)/quantaas-backend-service:latest