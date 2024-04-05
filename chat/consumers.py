import json
from channels.generic.websocket import WebsocketConsumer, AsyncWebsocketConsumer  # type: ignore


class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.room_group_name = "Test-Room"

        await self.channel_layer.group_add(self.room_group_name, self.channel_name)
        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.room_group_name, self.channel_name)
        print("Disconnected!")

    async def receive(self, text_data):
        receive_dict = json.loads(text_data)
        message = receive_dict["message"]
        print(message)
        action = receive_dict["action"]
        if action == "new-offer" or action == "new-answer":
            receiver_channel_name = receive_dict["message"]["receiver_channel_name"]
            receive_dict["message"]["receiver_channel_name"] = self.channel_name

            await self.channel_layer.send(
                receiver_channel_name,
                {"type": "send.sdp", "receive_dict": receive_dict},
            )
            return

        receive_dict["message"]["receiver_channel_name"] = self.channel_name

        await self.channel_layer.group_send(
            self.room_group_name, {"type": "send.sdp", "receive_dict": receive_dict}
        )

    async def sdp(self, event):
        receive_dict = event["receive_dict"]

        await self.send(text_data=json.dumps(receive_dict))


class SignalingConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()
        print("WebSocket connection established")

    def disconnect(self, close_code):
        print(f"WebSocket connection closed with code: {close_code}")

    def receive(self, text_data=None, bytes_data=None):
        print("Received signal")
        data = json.loads(text_data)

        # Handle WebRTC signaling messages
        if data.get("type") == "offer":
            self.handle_offer(data)
        elif data.get("type") == "answer":
            self.handle_answer(data)
        elif data.get("type") == "ice-candidate":
            self.handle_ice_candidate(data)
        else:
            print("Unknown message type:", data.get("type"))

    def handle_offer(self, offer):
        print("Received offer:", offer)

        # Normally, you would store the offer and send it to the other peer
        # Here, for simplicity, we just echo it back
        self.send(
            text_data=json.dumps({"type": offer["type"], "sdp": offer["offer"]["sdp"]})
        )

    def handle_answer(self, answer):
        print("Received answer:", answer)

        print("Sending answer:", answer)

        # Similarly, process and/or forward the answer to the initiating peer
        self.send(
            text_data=json.dumps(
                {"type": answer["type"], "sdp": answer["answer"]["sdp"]}
            )
        )

    def handle_ice_candidate(self, ice_candidate):
        print("Received ICE candidate:", ice_candidate)
        # ICE candidates are usually forwarded to the other peer
        self.send(
            text_data=json.dumps(
                {
                    "type": ice_candidate["type"],
                    "candidate": ice_candidate["candidate"]["candidate"],
                    "sdpMLineIndex": ice_candidate["candidate"]["sdpMLineIndex"],
                    "sdpMid": ice_candidate["candidate"]["sdpMid"],
                }
            )
        )
