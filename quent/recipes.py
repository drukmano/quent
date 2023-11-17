from quent import Chain


TrueChain = Chain().then(True).freeze()
FalseChain = Chain().then(False).freeze()
NoneChain = Chain().then(None).freeze()
LengthChain = Chain().then(len).freeze()
BoolChain = Chain().then(bool).freeze()
NotChain = Chain().not_().freeze()
