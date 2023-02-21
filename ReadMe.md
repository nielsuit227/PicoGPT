# PicoGPT
This was build after inspiration of Andrej Karpathy, who made a brilliant video on this transformer architecture. [^1]
It builds up nicely, starting with a simple self-attention head. 
Multiple attention heads are added sequentially, with an MLP after the heads. 
The residual connections are made, which improve performance drastically. 
An additional hidden layer after the heads and MLP again improve performance significantly.
The implementation of batch and layer norm is shortly touched, and PyTorch's implementation of the layer norm is added. 
Lastly, dropout is added, and then the network is scaled with parameters.

[^1]: https://www.youtube.com/watch?v=kCc8FmEb1nY


## Hardware limitation
Developing with a macbook leaves me a bit behind when it comes to scaling this network. 
It would be a great exercise to train this on the cloud, connecting with ClearML, W&B or Neptune and really try to scale this. Finally make the son of Anton come alive.

## Decoder only
Currently, this is a decoder-only network. It's purely generative, without any preconditioning of the output. For prompt answering or machine translations, the encoder part needs to be added and trained. 
This also requires some harder to acquire question / answer data. ChatGPT went through various cycles of finetuning on such (publically unavailable) data, optimizing an obscure reward policy. 
