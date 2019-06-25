from slacker import Slacker
import json
from flask import Flask, request, make_response

token = "xoxb-674813246676-674817950644-lfG8UTT9uYCcGnVlFQ9oNceO"
slack = Slacker(token)

app = Flask(__name__)

def get_answer():
    return "이거 보내지면 설정 된거에요 히히힣"

def event_handler(event_type, slack_event):
    if event_type == "app_mention":
        channel = slack_event["event"]["channel"]
        text = get_answer()
        slack.chat.post_message(channel, text)
        return make_response("앱 멘션 메시지가 보내졌습니다.",200,)
    message = "[%s] 이벤트 핸들러를 찾을 수 없습니다." % event_type


@app.route("/slack", methods=["GET","POST"])
def header(): # slack connect check
    slack_event = json.loads(request.data)
    if "challenge" in slack_event:
        return make_response(slack_event["challenge"], 200, {"content_type": "application/json"})

    if "event" in slack_event:
        event_type = slack_event["event"]["type"]
        return event_handler(event_type, slack_event)
    return make_response("슬랙 요청에 이벤트가 없습니다.", 404, {"X-Slack-No-Retry":1})

@app.route("/", methods=["GET","POST"])
def index():
    return "테스트"

if __name__ == '__main__':
    app.run('0.0.0.0', port=3030)