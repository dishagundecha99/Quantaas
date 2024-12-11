VERSION=v1
DOCKERUSER=dishagundecha99

build:
	docker build -f Dockerfile -t quantaas-job-status-service .

push:
	docker tag quantaas-job-status-service $(DOCKERUSER)/quantaas-job-status-service:$(VERSION)
	docker push $(DOCKERUSER)/quantaas-job-status-service:$(VERSION)
	docker tag quantaas-job-status-service $(DOCKERUSER)/quantaas-job-status-service:latest
	docker push $(DOCKERUSER)/quantaas-job-status-service:latest