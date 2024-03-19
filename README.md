# Node Classification on Cora Dataset using Graph Neural Network in Pytorch
Node Classification on large Knowledge Graphs of Cora Dataset using Graph Neural Network (GNN) in Pytorch.

Dataset:  Cora Dataset consists of 2708 scientific publications classified into one of seven classes.

#### File Structure
```
|-- data/
|-- weights/
|-- result/
|-- main.py
|-- gnn_model.py
|-- utils.py
|-- requirements.txt
```

### Training
```python
python3 main.py --mode train --epoch 2000
```
![train image](https://github.com/mr-ravin/GNN-Node-Classification-Cora-Pytorch/blob/main/result/training_analysis.png?raw=true)

### Testing
```python
python3 main.py --mode test
```
`Test Accuracy: 0.7970`

## License 
```
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
