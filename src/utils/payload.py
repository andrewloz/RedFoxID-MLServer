# what sort of structure or order do we want to recieve data in.
# do we recieve another header that specifies the field name?
# and we just map the message after that content header to a predict dict
# of somekind. 

# look up what a binary frame is, with length prefixes, avoid text, json and base64
import struct

class Payload:
    def __init__(self, message, imageContent, opcode = 1):
        self.message = message
        self.imageContent = imageContent
        self.opcode = opcode

    def to_frame(self):
        pass


    def from_frame_binary(self):
        pass