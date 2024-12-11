VERSION=v1
DOCKERUSER=dishagundecha99

build:
	docker build -f Dockerfile -t hyptaas-backend-service .

push:
	docker tag hyptaas-backend-service $(DOCKERUSER)/hyptaas-backend-service:$(VERSION)
	docker push $(DOCKERUSER)/hyptaas-backend-service:$(VERSION)
	docker tag hyptaas-backend-service $(DOCKERUSER)/hyptaas-backend-service:latest
	docker push $(DOCKERUSER)/hyptaas-backend-service:latest