def handler(request):
    if request.method == "GET":
        return {
            "status": "API jalan"
        }
