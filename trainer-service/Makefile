VERSION=v1
DOCKERUSER=dishagundecha99

build:
	docker build -f Dockerfile -t hyptaas-trainer-service .

push:
	docker tag hyptaas-trainer-service $(DOCKERUSER)/hyptaas-trainer-service:$(VERSION)
	docker push $(DOCKERUSER)/hyptaas-trainer-service:$(VERSION)
	docker tag hyptaas-trainer-service $(DOCKERUSER)/hyptaas-trainer-service:latest
	docker push $(DOCKERUSER)/hyptaas-trainer-service:latest