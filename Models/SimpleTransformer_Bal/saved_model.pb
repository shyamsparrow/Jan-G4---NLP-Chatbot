??!
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
R
Einsum
inputs"T*N
output"T"
equationstring"
Nint(0"	
Ttype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:
2		
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.7.02v2.7.0-0-gc256c071bb28??
{
dense_54/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_54/kernel
t
#dense_54/kernel/Read/ReadVariableOpReadVariableOpdense_54/kernel*
_output_shapes
:	?*
dtype0
r
dense_54/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_54/bias
k
!dense_54/bias/Read/ReadVariableOpReadVariableOpdense_54/bias*
_output_shapes
:*
dtype0
z
dense_55/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_55/kernel
s
#dense_55/kernel/Read/ReadVariableOpReadVariableOpdense_55/kernel*
_output_shapes

:*
dtype0
r
dense_55/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_55/bias
k
!dense_55/bias/Read/ReadVariableOpReadVariableOpdense_55/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
7token_and_position_embedding_13/embedding_26/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?)?*H
shared_name97token_and_position_embedding_13/embedding_26/embeddings
?
Ktoken_and_position_embedding_13/embedding_26/embeddings/Read/ReadVariableOpReadVariableOp7token_and_position_embedding_13/embedding_26/embeddings* 
_output_shapes
:
?)?*
dtype0
?
7token_and_position_embedding_13/embedding_27/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*H
shared_name97token_and_position_embedding_13/embedding_27/embeddings
?
Ktoken_and_position_embedding_13/embedding_27/embeddings/Read/ReadVariableOpReadVariableOp7token_and_position_embedding_13/embedding_27/embeddings* 
_output_shapes
:
??*
dtype0
?
9transformer_block_13/multi_head_attention_13/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*J
shared_name;9transformer_block_13/multi_head_attention_13/query/kernel
?
Mtransformer_block_13/multi_head_attention_13/query/kernel/Read/ReadVariableOpReadVariableOp9transformer_block_13/multi_head_attention_13/query/kernel*$
_output_shapes
:??*
dtype0
?
7transformer_block_13/multi_head_attention_13/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*H
shared_name97transformer_block_13/multi_head_attention_13/query/bias
?
Ktransformer_block_13/multi_head_attention_13/query/bias/Read/ReadVariableOpReadVariableOp7transformer_block_13/multi_head_attention_13/query/bias*
_output_shapes
:	?*
dtype0
?
7transformer_block_13/multi_head_attention_13/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*H
shared_name97transformer_block_13/multi_head_attention_13/key/kernel
?
Ktransformer_block_13/multi_head_attention_13/key/kernel/Read/ReadVariableOpReadVariableOp7transformer_block_13/multi_head_attention_13/key/kernel*$
_output_shapes
:??*
dtype0
?
5transformer_block_13/multi_head_attention_13/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*F
shared_name75transformer_block_13/multi_head_attention_13/key/bias
?
Itransformer_block_13/multi_head_attention_13/key/bias/Read/ReadVariableOpReadVariableOp5transformer_block_13/multi_head_attention_13/key/bias*
_output_shapes
:	?*
dtype0
?
9transformer_block_13/multi_head_attention_13/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*J
shared_name;9transformer_block_13/multi_head_attention_13/value/kernel
?
Mtransformer_block_13/multi_head_attention_13/value/kernel/Read/ReadVariableOpReadVariableOp9transformer_block_13/multi_head_attention_13/value/kernel*$
_output_shapes
:??*
dtype0
?
7transformer_block_13/multi_head_attention_13/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*H
shared_name97transformer_block_13/multi_head_attention_13/value/bias
?
Ktransformer_block_13/multi_head_attention_13/value/bias/Read/ReadVariableOpReadVariableOp7transformer_block_13/multi_head_attention_13/value/bias*
_output_shapes
:	?*
dtype0
?
Dtransformer_block_13/multi_head_attention_13/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*U
shared_nameFDtransformer_block_13/multi_head_attention_13/attention_output/kernel
?
Xtransformer_block_13/multi_head_attention_13/attention_output/kernel/Read/ReadVariableOpReadVariableOpDtransformer_block_13/multi_head_attention_13/attention_output/kernel*$
_output_shapes
:??*
dtype0
?
Btransformer_block_13/multi_head_attention_13/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*S
shared_nameDBtransformer_block_13/multi_head_attention_13/attention_output/bias
?
Vtransformer_block_13/multi_head_attention_13/attention_output/bias/Read/ReadVariableOpReadVariableOpBtransformer_block_13/multi_head_attention_13/attention_output/bias*
_output_shapes	
:?*
dtype0
|
dense_52/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_52/kernel
u
#dense_52/kernel/Read/ReadVariableOpReadVariableOpdense_52/kernel* 
_output_shapes
:
??*
dtype0
s
dense_52/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_52/bias
l
!dense_52/bias/Read/ReadVariableOpReadVariableOpdense_52/bias*
_output_shapes	
:?*
dtype0
|
dense_53/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_53/kernel
u
#dense_53/kernel/Read/ReadVariableOpReadVariableOpdense_53/kernel* 
_output_shapes
:
??*
dtype0
s
dense_53/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_53/bias
l
!dense_53/bias/Read/ReadVariableOpReadVariableOpdense_53/bias*
_output_shapes	
:?*
dtype0
?
1transformer_block_13/layer_normalization_26/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*B
shared_name31transformer_block_13/layer_normalization_26/gamma
?
Etransformer_block_13/layer_normalization_26/gamma/Read/ReadVariableOpReadVariableOp1transformer_block_13/layer_normalization_26/gamma*
_output_shapes	
:?*
dtype0
?
0transformer_block_13/layer_normalization_26/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*A
shared_name20transformer_block_13/layer_normalization_26/beta
?
Dtransformer_block_13/layer_normalization_26/beta/Read/ReadVariableOpReadVariableOp0transformer_block_13/layer_normalization_26/beta*
_output_shapes	
:?*
dtype0
?
1transformer_block_13/layer_normalization_27/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*B
shared_name31transformer_block_13/layer_normalization_27/gamma
?
Etransformer_block_13/layer_normalization_27/gamma/Read/ReadVariableOpReadVariableOp1transformer_block_13/layer_normalization_27/gamma*
_output_shapes	
:?*
dtype0
?
0transformer_block_13/layer_normalization_27/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*A
shared_name20transformer_block_13/layer_normalization_27/beta
?
Dtransformer_block_13/layer_normalization_27/beta/Read/ReadVariableOpReadVariableOp0transformer_block_13/layer_normalization_27/beta*
_output_shapes	
:?*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/dense_54/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_54/kernel/m
?
*Adam/dense_54/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_54/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_54/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_54/bias/m
y
(Adam/dense_54/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_54/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_55/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_55/kernel/m
?
*Adam/dense_55/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_55/kernel/m*
_output_shapes

:*
dtype0
?
Adam/dense_55/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_55/bias/m
y
(Adam/dense_55/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_55/bias/m*
_output_shapes
:*
dtype0
?
>Adam/token_and_position_embedding_13/embedding_26/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?)?*O
shared_name@>Adam/token_and_position_embedding_13/embedding_26/embeddings/m
?
RAdam/token_and_position_embedding_13/embedding_26/embeddings/m/Read/ReadVariableOpReadVariableOp>Adam/token_and_position_embedding_13/embedding_26/embeddings/m* 
_output_shapes
:
?)?*
dtype0
?
>Adam/token_and_position_embedding_13/embedding_27/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*O
shared_name@>Adam/token_and_position_embedding_13/embedding_27/embeddings/m
?
RAdam/token_and_position_embedding_13/embedding_27/embeddings/m/Read/ReadVariableOpReadVariableOp>Adam/token_and_position_embedding_13/embedding_27/embeddings/m* 
_output_shapes
:
??*
dtype0
?
@Adam/transformer_block_13/multi_head_attention_13/query/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*Q
shared_nameB@Adam/transformer_block_13/multi_head_attention_13/query/kernel/m
?
TAdam/transformer_block_13/multi_head_attention_13/query/kernel/m/Read/ReadVariableOpReadVariableOp@Adam/transformer_block_13/multi_head_attention_13/query/kernel/m*$
_output_shapes
:??*
dtype0
?
>Adam/transformer_block_13/multi_head_attention_13/query/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*O
shared_name@>Adam/transformer_block_13/multi_head_attention_13/query/bias/m
?
RAdam/transformer_block_13/multi_head_attention_13/query/bias/m/Read/ReadVariableOpReadVariableOp>Adam/transformer_block_13/multi_head_attention_13/query/bias/m*
_output_shapes
:	?*
dtype0
?
>Adam/transformer_block_13/multi_head_attention_13/key/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*O
shared_name@>Adam/transformer_block_13/multi_head_attention_13/key/kernel/m
?
RAdam/transformer_block_13/multi_head_attention_13/key/kernel/m/Read/ReadVariableOpReadVariableOp>Adam/transformer_block_13/multi_head_attention_13/key/kernel/m*$
_output_shapes
:??*
dtype0
?
<Adam/transformer_block_13/multi_head_attention_13/key/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*M
shared_name><Adam/transformer_block_13/multi_head_attention_13/key/bias/m
?
PAdam/transformer_block_13/multi_head_attention_13/key/bias/m/Read/ReadVariableOpReadVariableOp<Adam/transformer_block_13/multi_head_attention_13/key/bias/m*
_output_shapes
:	?*
dtype0
?
@Adam/transformer_block_13/multi_head_attention_13/value/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*Q
shared_nameB@Adam/transformer_block_13/multi_head_attention_13/value/kernel/m
?
TAdam/transformer_block_13/multi_head_attention_13/value/kernel/m/Read/ReadVariableOpReadVariableOp@Adam/transformer_block_13/multi_head_attention_13/value/kernel/m*$
_output_shapes
:??*
dtype0
?
>Adam/transformer_block_13/multi_head_attention_13/value/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*O
shared_name@>Adam/transformer_block_13/multi_head_attention_13/value/bias/m
?
RAdam/transformer_block_13/multi_head_attention_13/value/bias/m/Read/ReadVariableOpReadVariableOp>Adam/transformer_block_13/multi_head_attention_13/value/bias/m*
_output_shapes
:	?*
dtype0
?
KAdam/transformer_block_13/multi_head_attention_13/attention_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*\
shared_nameMKAdam/transformer_block_13/multi_head_attention_13/attention_output/kernel/m
?
_Adam/transformer_block_13/multi_head_attention_13/attention_output/kernel/m/Read/ReadVariableOpReadVariableOpKAdam/transformer_block_13/multi_head_attention_13/attention_output/kernel/m*$
_output_shapes
:??*
dtype0
?
IAdam/transformer_block_13/multi_head_attention_13/attention_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*Z
shared_nameKIAdam/transformer_block_13/multi_head_attention_13/attention_output/bias/m
?
]Adam/transformer_block_13/multi_head_attention_13/attention_output/bias/m/Read/ReadVariableOpReadVariableOpIAdam/transformer_block_13/multi_head_attention_13/attention_output/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_52/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_52/kernel/m
?
*Adam/dense_52/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_52/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_52/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_52/bias/m
z
(Adam/dense_52/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_52/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_53/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_53/kernel/m
?
*Adam/dense_53/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_53/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_53/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_53/bias/m
z
(Adam/dense_53/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_53/bias/m*
_output_shapes	
:?*
dtype0
?
8Adam/transformer_block_13/layer_normalization_26/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*I
shared_name:8Adam/transformer_block_13/layer_normalization_26/gamma/m
?
LAdam/transformer_block_13/layer_normalization_26/gamma/m/Read/ReadVariableOpReadVariableOp8Adam/transformer_block_13/layer_normalization_26/gamma/m*
_output_shapes	
:?*
dtype0
?
7Adam/transformer_block_13/layer_normalization_26/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*H
shared_name97Adam/transformer_block_13/layer_normalization_26/beta/m
?
KAdam/transformer_block_13/layer_normalization_26/beta/m/Read/ReadVariableOpReadVariableOp7Adam/transformer_block_13/layer_normalization_26/beta/m*
_output_shapes	
:?*
dtype0
?
8Adam/transformer_block_13/layer_normalization_27/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*I
shared_name:8Adam/transformer_block_13/layer_normalization_27/gamma/m
?
LAdam/transformer_block_13/layer_normalization_27/gamma/m/Read/ReadVariableOpReadVariableOp8Adam/transformer_block_13/layer_normalization_27/gamma/m*
_output_shapes	
:?*
dtype0
?
7Adam/transformer_block_13/layer_normalization_27/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*H
shared_name97Adam/transformer_block_13/layer_normalization_27/beta/m
?
KAdam/transformer_block_13/layer_normalization_27/beta/m/Read/ReadVariableOpReadVariableOp7Adam/transformer_block_13/layer_normalization_27/beta/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_54/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_54/kernel/v
?
*Adam/dense_54/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_54/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/dense_54/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_54/bias/v
y
(Adam/dense_54/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_54/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_55/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_55/kernel/v
?
*Adam/dense_55/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_55/kernel/v*
_output_shapes

:*
dtype0
?
Adam/dense_55/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_55/bias/v
y
(Adam/dense_55/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_55/bias/v*
_output_shapes
:*
dtype0
?
>Adam/token_and_position_embedding_13/embedding_26/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?)?*O
shared_name@>Adam/token_and_position_embedding_13/embedding_26/embeddings/v
?
RAdam/token_and_position_embedding_13/embedding_26/embeddings/v/Read/ReadVariableOpReadVariableOp>Adam/token_and_position_embedding_13/embedding_26/embeddings/v* 
_output_shapes
:
?)?*
dtype0
?
>Adam/token_and_position_embedding_13/embedding_27/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*O
shared_name@>Adam/token_and_position_embedding_13/embedding_27/embeddings/v
?
RAdam/token_and_position_embedding_13/embedding_27/embeddings/v/Read/ReadVariableOpReadVariableOp>Adam/token_and_position_embedding_13/embedding_27/embeddings/v* 
_output_shapes
:
??*
dtype0
?
@Adam/transformer_block_13/multi_head_attention_13/query/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*Q
shared_nameB@Adam/transformer_block_13/multi_head_attention_13/query/kernel/v
?
TAdam/transformer_block_13/multi_head_attention_13/query/kernel/v/Read/ReadVariableOpReadVariableOp@Adam/transformer_block_13/multi_head_attention_13/query/kernel/v*$
_output_shapes
:??*
dtype0
?
>Adam/transformer_block_13/multi_head_attention_13/query/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*O
shared_name@>Adam/transformer_block_13/multi_head_attention_13/query/bias/v
?
RAdam/transformer_block_13/multi_head_attention_13/query/bias/v/Read/ReadVariableOpReadVariableOp>Adam/transformer_block_13/multi_head_attention_13/query/bias/v*
_output_shapes
:	?*
dtype0
?
>Adam/transformer_block_13/multi_head_attention_13/key/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*O
shared_name@>Adam/transformer_block_13/multi_head_attention_13/key/kernel/v
?
RAdam/transformer_block_13/multi_head_attention_13/key/kernel/v/Read/ReadVariableOpReadVariableOp>Adam/transformer_block_13/multi_head_attention_13/key/kernel/v*$
_output_shapes
:??*
dtype0
?
<Adam/transformer_block_13/multi_head_attention_13/key/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*M
shared_name><Adam/transformer_block_13/multi_head_attention_13/key/bias/v
?
PAdam/transformer_block_13/multi_head_attention_13/key/bias/v/Read/ReadVariableOpReadVariableOp<Adam/transformer_block_13/multi_head_attention_13/key/bias/v*
_output_shapes
:	?*
dtype0
?
@Adam/transformer_block_13/multi_head_attention_13/value/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*Q
shared_nameB@Adam/transformer_block_13/multi_head_attention_13/value/kernel/v
?
TAdam/transformer_block_13/multi_head_attention_13/value/kernel/v/Read/ReadVariableOpReadVariableOp@Adam/transformer_block_13/multi_head_attention_13/value/kernel/v*$
_output_shapes
:??*
dtype0
?
>Adam/transformer_block_13/multi_head_attention_13/value/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*O
shared_name@>Adam/transformer_block_13/multi_head_attention_13/value/bias/v
?
RAdam/transformer_block_13/multi_head_attention_13/value/bias/v/Read/ReadVariableOpReadVariableOp>Adam/transformer_block_13/multi_head_attention_13/value/bias/v*
_output_shapes
:	?*
dtype0
?
KAdam/transformer_block_13/multi_head_attention_13/attention_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*\
shared_nameMKAdam/transformer_block_13/multi_head_attention_13/attention_output/kernel/v
?
_Adam/transformer_block_13/multi_head_attention_13/attention_output/kernel/v/Read/ReadVariableOpReadVariableOpKAdam/transformer_block_13/multi_head_attention_13/attention_output/kernel/v*$
_output_shapes
:??*
dtype0
?
IAdam/transformer_block_13/multi_head_attention_13/attention_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*Z
shared_nameKIAdam/transformer_block_13/multi_head_attention_13/attention_output/bias/v
?
]Adam/transformer_block_13/multi_head_attention_13/attention_output/bias/v/Read/ReadVariableOpReadVariableOpIAdam/transformer_block_13/multi_head_attention_13/attention_output/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_52/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_52/kernel/v
?
*Adam/dense_52/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_52/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_52/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_52/bias/v
z
(Adam/dense_52/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_52/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_53/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_53/kernel/v
?
*Adam/dense_53/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_53/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_53/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_53/bias/v
z
(Adam/dense_53/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_53/bias/v*
_output_shapes	
:?*
dtype0
?
8Adam/transformer_block_13/layer_normalization_26/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*I
shared_name:8Adam/transformer_block_13/layer_normalization_26/gamma/v
?
LAdam/transformer_block_13/layer_normalization_26/gamma/v/Read/ReadVariableOpReadVariableOp8Adam/transformer_block_13/layer_normalization_26/gamma/v*
_output_shapes	
:?*
dtype0
?
7Adam/transformer_block_13/layer_normalization_26/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*H
shared_name97Adam/transformer_block_13/layer_normalization_26/beta/v
?
KAdam/transformer_block_13/layer_normalization_26/beta/v/Read/ReadVariableOpReadVariableOp7Adam/transformer_block_13/layer_normalization_26/beta/v*
_output_shapes	
:?*
dtype0
?
8Adam/transformer_block_13/layer_normalization_27/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*I
shared_name:8Adam/transformer_block_13/layer_normalization_27/gamma/v
?
LAdam/transformer_block_13/layer_normalization_27/gamma/v/Read/ReadVariableOpReadVariableOp8Adam/transformer_block_13/layer_normalization_27/gamma/v*
_output_shapes	
:?*
dtype0
?
7Adam/transformer_block_13/layer_normalization_27/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*H
shared_name97Adam/transformer_block_13/layer_normalization_27/beta/v
?
KAdam/transformer_block_13/layer_normalization_27/beta/v/Read/ReadVariableOpReadVariableOp7Adam/transformer_block_13/layer_normalization_27/beta/v*
_output_shapes	
:?*
dtype0

NoOpNoOp
Օ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
n
	token_emb
pos_emb
	variables
trainable_variables
regularization_losses
	keras_api
?
att
ffn

layernorm1

layernorm2
dropout1
dropout2
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
 trainable_variables
!regularization_losses
"	keras_api
R
#	variables
$trainable_variables
%regularization_losses
&	keras_api
h

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
R
-	variables
.trainable_variables
/regularization_losses
0	keras_api
h

1kernel
2bias
3	variables
4trainable_variables
5regularization_losses
6	keras_api
?
7iter

8beta_1

9beta_2
	:decay
;learning_rate'm?(m?1m?2m?<m?=m?>m??m?@m?Am?Bm?Cm?Dm?Em?Fm?Gm?Hm?Im?Jm?Km?Lm?Mm?'v?(v?1v?2v?<v?=v?>v??v?@v?Av?Bv?Cv?Dv?Ev?Fv?Gv?Hv?Iv?Jv?Kv?Lv?Mv?
?
<0
=1
>2
?3
@4
A5
B6
C7
D8
E9
F10
G11
H12
I13
J14
K15
L16
M17
'18
(19
120
221
?
<0
=1
>2
?3
@4
A5
B6
C7
D8
E9
F10
G11
H12
I13
J14
K15
L16
M17
'18
(19
120
221
 
?
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics

	variables
trainable_variables
regularization_losses
 
b
<
embeddings
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
b
=
embeddings
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api

<0
=1

<0
=1
 
?
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
	variables
trainable_variables
regularization_losses
?
`_query_dense
a
_key_dense
b_value_dense
c_softmax
d_dropout_layer
e_output_dense
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
?
jlayer_with_weights-0
jlayer-0
klayer_with_weights-1
klayer-1
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
q
paxis
	Jgamma
Kbeta
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
q
uaxis
	Lgamma
Mbeta
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
R
z	variables
{trainable_variables
|regularization_losses
}	keras_api
T
~	variables
trainable_variables
?regularization_losses
?	keras_api
v
>0
?1
@2
A3
B4
C5
D6
E7
F8
G9
H10
I11
J12
K13
L14
M15
v
>0
?1
@2
A3
B4
C5
D6
E7
F8
G9
H10
I11
J12
K13
L14
M15
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
 trainable_variables
!regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
#	variables
$trainable_variables
%regularization_losses
[Y
VARIABLE_VALUEdense_54/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_54/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

'0
(1

'0
(1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
)	variables
*trainable_variables
+regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
-	variables
.trainable_variables
/regularization_losses
[Y
VARIABLE_VALUEdense_55/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_55/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

10
21

10
21
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
3	variables
4trainable_variables
5regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE7token_and_position_embedding_13/embedding_26/embeddings&variables/0/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE7token_and_position_embedding_13/embedding_27/embeddings&variables/1/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE9transformer_block_13/multi_head_attention_13/query/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE7transformer_block_13/multi_head_attention_13/query/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE7transformer_block_13/multi_head_attention_13/key/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE5transformer_block_13/multi_head_attention_13/key/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE9transformer_block_13/multi_head_attention_13/value/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE7transformer_block_13/multi_head_attention_13/value/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEDtransformer_block_13/multi_head_attention_13/attention_output/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEBtransformer_block_13/multi_head_attention_13/attention_output/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_52/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_52/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_53/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_53/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE1transformer_block_13/layer_normalization_26/gamma'variables/14/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE0transformer_block_13/layer_normalization_26/beta'variables/15/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE1transformer_block_13/layer_normalization_27/gamma'variables/16/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE0transformer_block_13/layer_normalization_27/beta'variables/17/.ATTRIBUTES/VARIABLE_VALUE
 
8
0
1
2
3
4
5
6
7

?0
?1
 
 

<0

<0
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses

=0

=0
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
 

0
1
 
 
 
?
?partial_output_shape
?full_output_shape

>kernel
?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
?partial_output_shape
?full_output_shape

@kernel
Abias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
?partial_output_shape
?full_output_shape

Bkernel
Cbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
?partial_output_shape
?full_output_shape

Dkernel
Ebias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
8
>0
?1
@2
A3
B4
C5
D6
E7
8
>0
?1
@2
A3
B4
C5
D6
E7
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
f	variables
gtrainable_variables
hregularization_losses
l

Fkernel
Gbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
l

Hkernel
Ibias
?	variables
?trainable_variables
?regularization_losses
?	keras_api

F0
G1
H2
I3

F0
G1
H2
I3
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
l	variables
mtrainable_variables
nregularization_losses
 

J0
K1

J0
K1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
q	variables
rtrainable_variables
sregularization_losses
 

L0
M1

L0
M1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
v	variables
wtrainable_variables
xregularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
z	variables
{trainable_variables
|regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
~	variables
trainable_variables
?regularization_losses
 
*
0
1
2
3
4
5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
 
 
 
 
 
 
 
 
 
 
 
 

>0
?1

>0
?1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 

@0
A1

@0
A1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 

B0
C1

B0
C1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 

D0
E1

D0
E1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
*
`0
a1
b2
c3
d4
e5
 
 
 

F0
G1

F0
G1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses

H0
I1

H0
I1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 

j0
k1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
~|
VARIABLE_VALUEAdam/dense_54/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_54/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_55/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_55/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE>Adam/token_and_position_embedding_13/embedding_26/embeddings/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE>Adam/token_and_position_embedding_13/embedding_27/embeddings/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@Adam/transformer_block_13/multi_head_attention_13/query/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE>Adam/transformer_block_13/multi_head_attention_13/query/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE>Adam/transformer_block_13/multi_head_attention_13/key/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE<Adam/transformer_block_13/multi_head_attention_13/key/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@Adam/transformer_block_13/multi_head_attention_13/value/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE>Adam/transformer_block_13/multi_head_attention_13/value/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEKAdam/transformer_block_13/multi_head_attention_13/attention_output/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/transformer_block_13/multi_head_attention_13/attention_output/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_52/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_52/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_53/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_53/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/transformer_block_13/layer_normalization_26/gamma/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE7Adam/transformer_block_13/layer_normalization_26/beta/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/transformer_block_13/layer_normalization_27/gamma/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE7Adam/transformer_block_13/layer_normalization_27/beta/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_54/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_54/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_55/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_55/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE>Adam/token_and_position_embedding_13/embedding_26/embeddings/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE>Adam/token_and_position_embedding_13/embedding_27/embeddings/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@Adam/transformer_block_13/multi_head_attention_13/query/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE>Adam/transformer_block_13/multi_head_attention_13/query/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE>Adam/transformer_block_13/multi_head_attention_13/key/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE<Adam/transformer_block_13/multi_head_attention_13/key/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@Adam/transformer_block_13/multi_head_attention_13/value/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE>Adam/transformer_block_13/multi_head_attention_13/value/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEKAdam/transformer_block_13/multi_head_attention_13/attention_output/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/transformer_block_13/multi_head_attention_13/attention_output/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_52/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_52/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_53/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_53/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/transformer_block_13/layer_normalization_26/gamma/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE7Adam/transformer_block_13/layer_normalization_26/beta/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/transformer_block_13/layer_normalization_27/gamma/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE7Adam/transformer_block_13/layer_normalization_27/beta/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
serving_default_input_14Placeholder*(
_output_shapes
:??????????*
dtype0*
shape:??????????
?	
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_147token_and_position_embedding_13/embedding_27/embeddings7token_and_position_embedding_13/embedding_26/embeddings9transformer_block_13/multi_head_attention_13/query/kernel7transformer_block_13/multi_head_attention_13/query/bias7transformer_block_13/multi_head_attention_13/key/kernel5transformer_block_13/multi_head_attention_13/key/bias9transformer_block_13/multi_head_attention_13/value/kernel7transformer_block_13/multi_head_attention_13/value/biasDtransformer_block_13/multi_head_attention_13/attention_output/kernelBtransformer_block_13/multi_head_attention_13/attention_output/bias1transformer_block_13/layer_normalization_26/gamma0transformer_block_13/layer_normalization_26/betadense_52/kerneldense_52/biasdense_53/kerneldense_53/bias1transformer_block_13/layer_normalization_27/gamma0transformer_block_13/layer_normalization_27/betadense_54/kerneldense_54/biasdense_55/kerneldense_55/bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_82441
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?'
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_54/kernel/Read/ReadVariableOp!dense_54/bias/Read/ReadVariableOp#dense_55/kernel/Read/ReadVariableOp!dense_55/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpKtoken_and_position_embedding_13/embedding_26/embeddings/Read/ReadVariableOpKtoken_and_position_embedding_13/embedding_27/embeddings/Read/ReadVariableOpMtransformer_block_13/multi_head_attention_13/query/kernel/Read/ReadVariableOpKtransformer_block_13/multi_head_attention_13/query/bias/Read/ReadVariableOpKtransformer_block_13/multi_head_attention_13/key/kernel/Read/ReadVariableOpItransformer_block_13/multi_head_attention_13/key/bias/Read/ReadVariableOpMtransformer_block_13/multi_head_attention_13/value/kernel/Read/ReadVariableOpKtransformer_block_13/multi_head_attention_13/value/bias/Read/ReadVariableOpXtransformer_block_13/multi_head_attention_13/attention_output/kernel/Read/ReadVariableOpVtransformer_block_13/multi_head_attention_13/attention_output/bias/Read/ReadVariableOp#dense_52/kernel/Read/ReadVariableOp!dense_52/bias/Read/ReadVariableOp#dense_53/kernel/Read/ReadVariableOp!dense_53/bias/Read/ReadVariableOpEtransformer_block_13/layer_normalization_26/gamma/Read/ReadVariableOpDtransformer_block_13/layer_normalization_26/beta/Read/ReadVariableOpEtransformer_block_13/layer_normalization_27/gamma/Read/ReadVariableOpDtransformer_block_13/layer_normalization_27/beta/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_54/kernel/m/Read/ReadVariableOp(Adam/dense_54/bias/m/Read/ReadVariableOp*Adam/dense_55/kernel/m/Read/ReadVariableOp(Adam/dense_55/bias/m/Read/ReadVariableOpRAdam/token_and_position_embedding_13/embedding_26/embeddings/m/Read/ReadVariableOpRAdam/token_and_position_embedding_13/embedding_27/embeddings/m/Read/ReadVariableOpTAdam/transformer_block_13/multi_head_attention_13/query/kernel/m/Read/ReadVariableOpRAdam/transformer_block_13/multi_head_attention_13/query/bias/m/Read/ReadVariableOpRAdam/transformer_block_13/multi_head_attention_13/key/kernel/m/Read/ReadVariableOpPAdam/transformer_block_13/multi_head_attention_13/key/bias/m/Read/ReadVariableOpTAdam/transformer_block_13/multi_head_attention_13/value/kernel/m/Read/ReadVariableOpRAdam/transformer_block_13/multi_head_attention_13/value/bias/m/Read/ReadVariableOp_Adam/transformer_block_13/multi_head_attention_13/attention_output/kernel/m/Read/ReadVariableOp]Adam/transformer_block_13/multi_head_attention_13/attention_output/bias/m/Read/ReadVariableOp*Adam/dense_52/kernel/m/Read/ReadVariableOp(Adam/dense_52/bias/m/Read/ReadVariableOp*Adam/dense_53/kernel/m/Read/ReadVariableOp(Adam/dense_53/bias/m/Read/ReadVariableOpLAdam/transformer_block_13/layer_normalization_26/gamma/m/Read/ReadVariableOpKAdam/transformer_block_13/layer_normalization_26/beta/m/Read/ReadVariableOpLAdam/transformer_block_13/layer_normalization_27/gamma/m/Read/ReadVariableOpKAdam/transformer_block_13/layer_normalization_27/beta/m/Read/ReadVariableOp*Adam/dense_54/kernel/v/Read/ReadVariableOp(Adam/dense_54/bias/v/Read/ReadVariableOp*Adam/dense_55/kernel/v/Read/ReadVariableOp(Adam/dense_55/bias/v/Read/ReadVariableOpRAdam/token_and_position_embedding_13/embedding_26/embeddings/v/Read/ReadVariableOpRAdam/token_and_position_embedding_13/embedding_27/embeddings/v/Read/ReadVariableOpTAdam/transformer_block_13/multi_head_attention_13/query/kernel/v/Read/ReadVariableOpRAdam/transformer_block_13/multi_head_attention_13/query/bias/v/Read/ReadVariableOpRAdam/transformer_block_13/multi_head_attention_13/key/kernel/v/Read/ReadVariableOpPAdam/transformer_block_13/multi_head_attention_13/key/bias/v/Read/ReadVariableOpTAdam/transformer_block_13/multi_head_attention_13/value/kernel/v/Read/ReadVariableOpRAdam/transformer_block_13/multi_head_attention_13/value/bias/v/Read/ReadVariableOp_Adam/transformer_block_13/multi_head_attention_13/attention_output/kernel/v/Read/ReadVariableOp]Adam/transformer_block_13/multi_head_attention_13/attention_output/bias/v/Read/ReadVariableOp*Adam/dense_52/kernel/v/Read/ReadVariableOp(Adam/dense_52/bias/v/Read/ReadVariableOp*Adam/dense_53/kernel/v/Read/ReadVariableOp(Adam/dense_53/bias/v/Read/ReadVariableOpLAdam/transformer_block_13/layer_normalization_26/gamma/v/Read/ReadVariableOpKAdam/transformer_block_13/layer_normalization_26/beta/v/Read/ReadVariableOpLAdam/transformer_block_13/layer_normalization_27/gamma/v/Read/ReadVariableOpKAdam/transformer_block_13/layer_normalization_27/beta/v/Read/ReadVariableOpConst*X
TinQ
O2M	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_84085
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_54/kerneldense_54/biasdense_55/kerneldense_55/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate7token_and_position_embedding_13/embedding_26/embeddings7token_and_position_embedding_13/embedding_27/embeddings9transformer_block_13/multi_head_attention_13/query/kernel7transformer_block_13/multi_head_attention_13/query/bias7transformer_block_13/multi_head_attention_13/key/kernel5transformer_block_13/multi_head_attention_13/key/bias9transformer_block_13/multi_head_attention_13/value/kernel7transformer_block_13/multi_head_attention_13/value/biasDtransformer_block_13/multi_head_attention_13/attention_output/kernelBtransformer_block_13/multi_head_attention_13/attention_output/biasdense_52/kerneldense_52/biasdense_53/kerneldense_53/bias1transformer_block_13/layer_normalization_26/gamma0transformer_block_13/layer_normalization_26/beta1transformer_block_13/layer_normalization_27/gamma0transformer_block_13/layer_normalization_27/betatotalcounttotal_1count_1Adam/dense_54/kernel/mAdam/dense_54/bias/mAdam/dense_55/kernel/mAdam/dense_55/bias/m>Adam/token_and_position_embedding_13/embedding_26/embeddings/m>Adam/token_and_position_embedding_13/embedding_27/embeddings/m@Adam/transformer_block_13/multi_head_attention_13/query/kernel/m>Adam/transformer_block_13/multi_head_attention_13/query/bias/m>Adam/transformer_block_13/multi_head_attention_13/key/kernel/m<Adam/transformer_block_13/multi_head_attention_13/key/bias/m@Adam/transformer_block_13/multi_head_attention_13/value/kernel/m>Adam/transformer_block_13/multi_head_attention_13/value/bias/mKAdam/transformer_block_13/multi_head_attention_13/attention_output/kernel/mIAdam/transformer_block_13/multi_head_attention_13/attention_output/bias/mAdam/dense_52/kernel/mAdam/dense_52/bias/mAdam/dense_53/kernel/mAdam/dense_53/bias/m8Adam/transformer_block_13/layer_normalization_26/gamma/m7Adam/transformer_block_13/layer_normalization_26/beta/m8Adam/transformer_block_13/layer_normalization_27/gamma/m7Adam/transformer_block_13/layer_normalization_27/beta/mAdam/dense_54/kernel/vAdam/dense_54/bias/vAdam/dense_55/kernel/vAdam/dense_55/bias/v>Adam/token_and_position_embedding_13/embedding_26/embeddings/v>Adam/token_and_position_embedding_13/embedding_27/embeddings/v@Adam/transformer_block_13/multi_head_attention_13/query/kernel/v>Adam/transformer_block_13/multi_head_attention_13/query/bias/v>Adam/transformer_block_13/multi_head_attention_13/key/kernel/v<Adam/transformer_block_13/multi_head_attention_13/key/bias/v@Adam/transformer_block_13/multi_head_attention_13/value/kernel/v>Adam/transformer_block_13/multi_head_attention_13/value/bias/vKAdam/transformer_block_13/multi_head_attention_13/attention_output/kernel/vIAdam/transformer_block_13/multi_head_attention_13/attention_output/bias/vAdam/dense_52/kernel/vAdam/dense_52/bias/vAdam/dense_53/kernel/vAdam/dense_53/bias/v8Adam/transformer_block_13/layer_normalization_26/gamma/v7Adam/transformer_block_13/layer_normalization_26/beta/v8Adam/transformer_block_13/layer_normalization_27/gamma/v7Adam/transformer_block_13/layer_normalization_27/beta/v*W
TinP
N2L*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_84320??
?
?
(__inference_dense_55_layer_call_fn_83607

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_55_layer_call_and_return_conditional_losses_81664o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
H__inference_sequential_13_layer_call_and_return_conditional_losses_83701

inputs>
*dense_52_tensordot_readvariableop_resource:
??7
(dense_52_biasadd_readvariableop_resource:	?>
*dense_53_tensordot_readvariableop_resource:
??7
(dense_53_biasadd_readvariableop_resource:	?
identity??dense_52/BiasAdd/ReadVariableOp?!dense_52/Tensordot/ReadVariableOp?dense_53/BiasAdd/ReadVariableOp?!dense_53/Tensordot/ReadVariableOp?
!dense_52/Tensordot/ReadVariableOpReadVariableOp*dense_52_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype0a
dense_52/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_52/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       N
dense_52/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:b
 dense_52/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_52/Tensordot/GatherV2GatherV2!dense_52/Tensordot/Shape:output:0 dense_52/Tensordot/free:output:0)dense_52/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_52/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_52/Tensordot/GatherV2_1GatherV2!dense_52/Tensordot/Shape:output:0 dense_52/Tensordot/axes:output:0+dense_52/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_52/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_52/Tensordot/ProdProd$dense_52/Tensordot/GatherV2:output:0!dense_52/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_52/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_52/Tensordot/Prod_1Prod&dense_52/Tensordot/GatherV2_1:output:0#dense_52/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_52/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_52/Tensordot/concatConcatV2 dense_52/Tensordot/free:output:0 dense_52/Tensordot/axes:output:0'dense_52/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_52/Tensordot/stackPack dense_52/Tensordot/Prod:output:0"dense_52/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_52/Tensordot/transpose	Transposeinputs"dense_52/Tensordot/concat:output:0*
T0*-
_output_shapes
:????????????
dense_52/Tensordot/ReshapeReshape dense_52/Tensordot/transpose:y:0!dense_52/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_52/Tensordot/MatMulMatMul#dense_52/Tensordot/Reshape:output:0)dense_52/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????e
dense_52/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?b
 dense_52/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_52/Tensordot/concat_1ConcatV2$dense_52/Tensordot/GatherV2:output:0#dense_52/Tensordot/Const_2:output:0)dense_52/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_52/TensordotReshape#dense_52/Tensordot/MatMul:product:0$dense_52/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:????????????
dense_52/BiasAdd/ReadVariableOpReadVariableOp(dense_52_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_52/BiasAddBiasAdddense_52/Tensordot:output:0'dense_52/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????h
dense_52/ReluReludense_52/BiasAdd:output:0*
T0*-
_output_shapes
:????????????
!dense_53/Tensordot/ReadVariableOpReadVariableOp*dense_53_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype0a
dense_53/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_53/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       c
dense_53/Tensordot/ShapeShapedense_52/Relu:activations:0*
T0*
_output_shapes
:b
 dense_53/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_53/Tensordot/GatherV2GatherV2!dense_53/Tensordot/Shape:output:0 dense_53/Tensordot/free:output:0)dense_53/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_53/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_53/Tensordot/GatherV2_1GatherV2!dense_53/Tensordot/Shape:output:0 dense_53/Tensordot/axes:output:0+dense_53/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_53/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_53/Tensordot/ProdProd$dense_53/Tensordot/GatherV2:output:0!dense_53/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_53/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_53/Tensordot/Prod_1Prod&dense_53/Tensordot/GatherV2_1:output:0#dense_53/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_53/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_53/Tensordot/concatConcatV2 dense_53/Tensordot/free:output:0 dense_53/Tensordot/axes:output:0'dense_53/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_53/Tensordot/stackPack dense_53/Tensordot/Prod:output:0"dense_53/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_53/Tensordot/transpose	Transposedense_52/Relu:activations:0"dense_53/Tensordot/concat:output:0*
T0*-
_output_shapes
:????????????
dense_53/Tensordot/ReshapeReshape dense_53/Tensordot/transpose:y:0!dense_53/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_53/Tensordot/MatMulMatMul#dense_53/Tensordot/Reshape:output:0)dense_53/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????e
dense_53/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?b
 dense_53/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_53/Tensordot/concat_1ConcatV2$dense_53/Tensordot/GatherV2:output:0#dense_53/Tensordot/Const_2:output:0)dense_53/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_53/TensordotReshape#dense_53/Tensordot/MatMul:product:0$dense_53/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:????????????
dense_53/BiasAdd/ReadVariableOpReadVariableOp(dense_53_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_53/BiasAddBiasAdddense_53/Tensordot:output:0'dense_53/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????n
IdentityIdentitydense_53/BiasAdd:output:0^NoOp*
T0*-
_output_shapes
:????????????
NoOpNoOp ^dense_52/BiasAdd/ReadVariableOp"^dense_52/Tensordot/ReadVariableOp ^dense_53/BiasAdd/ReadVariableOp"^dense_53/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : : : 2B
dense_52/BiasAdd/ReadVariableOpdense_52/BiasAdd/ReadVariableOp2F
!dense_52/Tensordot/ReadVariableOp!dense_52/Tensordot/ReadVariableOp2B
dense_53/BiasAdd/ReadVariableOpdense_53/BiasAdd/ReadVariableOp2F
!dense_53/Tensordot/ReadVariableOp!dense_53/Tensordot/ReadVariableOp:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
-__inference_sequential_13_layer_call_fn_83631

inputs
unknown:
??
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_13_layer_call_and_return_conditional_losses_81234u
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
C__inference_dense_55_layer_call_and_return_conditional_losses_81664

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
F
*__inference_dropout_58_layer_call_fn_83576

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_58_layer_call_and_return_conditional_losses_81651`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
Z__inference_token_and_position_embedding_13_layer_call_and_return_conditional_losses_81390
x7
#embedding_27_embedding_lookup_81377:
??7
#embedding_26_embedding_lookup_81383:
?)?
identity??embedding_26/embedding_lookup?embedding_27/embedding_lookup6
ShapeShapex*
T0*
_output_shapes
:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :o
rangeRangerange/start:output:0strided_slice:output:0range/delta:output:0*
_output_shapes	
:??
embedding_27/embedding_lookupResourceGather#embedding_27_embedding_lookup_81377range:output:0*
Tindices0*6
_class,
*(loc:@embedding_27/embedding_lookup/81377* 
_output_shapes
:
??*
dtype0?
&embedding_27/embedding_lookup/IdentityIdentity&embedding_27/embedding_lookup:output:0*
T0*6
_class,
*(loc:@embedding_27/embedding_lookup/81377* 
_output_shapes
:
???
(embedding_27/embedding_lookup/Identity_1Identity/embedding_27/embedding_lookup/Identity:output:0*
T0* 
_output_shapes
:
??^
embedding_26/CastCastx*

DstT0*

SrcT0*(
_output_shapes
:???????????
embedding_26/embedding_lookupResourceGather#embedding_26_embedding_lookup_81383embedding_26/Cast:y:0*
Tindices0*6
_class,
*(loc:@embedding_26/embedding_lookup/81383*-
_output_shapes
:???????????*
dtype0?
&embedding_26/embedding_lookup/IdentityIdentity&embedding_26/embedding_lookup:output:0*
T0*6
_class,
*(loc:@embedding_26/embedding_lookup/81383*-
_output_shapes
:????????????
(embedding_26/embedding_lookup/Identity_1Identity/embedding_26/embedding_lookup/Identity:output:0*
T0*-
_output_shapes
:????????????
addAddV21embedding_26/embedding_lookup/Identity_1:output:01embedding_27/embedding_lookup/Identity_1:output:0*
T0*-
_output_shapes
:???????????\
IdentityIdentityadd:z:0^NoOp*
T0*-
_output_shapes
:????????????
NoOpNoOp^embedding_26/embedding_lookup^embedding_27/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2>
embedding_26/embedding_lookupembedding_26/embedding_lookup2>
embedding_27/embedding_lookupembedding_27/embedding_lookup:K G
(
_output_shapes
:??????????

_user_specified_namex
?
c
*__inference_dropout_58_layer_call_fn_83581

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_58_layer_call_and_return_conditional_losses_81748o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
r
V__inference_global_average_pooling1d_13_layer_call_and_return_conditional_losses_81356

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?.
?

C__inference_model_13_layer_call_and_return_conditional_losses_82384
input_149
%token_and_position_embedding_13_82332:
??9
%token_and_position_embedding_13_82334:
?)?2
transformer_block_13_82337:??-
transformer_block_13_82339:	?2
transformer_block_13_82341:??-
transformer_block_13_82343:	?2
transformer_block_13_82345:??-
transformer_block_13_82347:	?2
transformer_block_13_82349:??)
transformer_block_13_82351:	?)
transformer_block_13_82353:	?)
transformer_block_13_82355:	?.
transformer_block_13_82357:
??)
transformer_block_13_82359:	?.
transformer_block_13_82361:
??)
transformer_block_13_82363:	?)
transformer_block_13_82365:	?)
transformer_block_13_82367:	?!
dense_54_82372:	?
dense_54_82374: 
dense_55_82378:
dense_55_82380:
identity?? dense_54/StatefulPartitionedCall? dense_55/StatefulPartitionedCall?"dropout_57/StatefulPartitionedCall?"dropout_58/StatefulPartitionedCall?7token_and_position_embedding_13/StatefulPartitionedCall?,transformer_block_13/StatefulPartitionedCall?
7token_and_position_embedding_13/StatefulPartitionedCallStatefulPartitionedCallinput_14%token_and_position_embedding_13_82332%token_and_position_embedding_13_82334*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *c
f^R\
Z__inference_token_and_position_embedding_13_layer_call_and_return_conditional_losses_81390?
,transformer_block_13/StatefulPartitionedCallStatefulPartitionedCall@token_and_position_embedding_13/StatefulPartitionedCall:output:0transformer_block_13_82337transformer_block_13_82339transformer_block_13_82341transformer_block_13_82343transformer_block_13_82345transformer_block_13_82347transformer_block_13_82349transformer_block_13_82351transformer_block_13_82353transformer_block_13_82355transformer_block_13_82357transformer_block_13_82359transformer_block_13_82361transformer_block_13_82363transformer_block_13_82365transformer_block_13_82367*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_transformer_block_13_layer_call_and_return_conditional_losses_82027?
+global_average_pooling1d_13/PartitionedCallPartitionedCall5transformer_block_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *_
fZRX
V__inference_global_average_pooling1d_13_layer_call_and_return_conditional_losses_81620?
"dropout_57/StatefulPartitionedCallStatefulPartitionedCall4global_average_pooling1d_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_57_layer_call_and_return_conditional_losses_81781?
 dense_54/StatefulPartitionedCallStatefulPartitionedCall+dropout_57/StatefulPartitionedCall:output:0dense_54_82372dense_54_82374*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_54_layer_call_and_return_conditional_losses_81640?
"dropout_58/StatefulPartitionedCallStatefulPartitionedCall)dense_54/StatefulPartitionedCall:output:0#^dropout_57/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_58_layer_call_and_return_conditional_losses_81748?
 dense_55/StatefulPartitionedCallStatefulPartitionedCall+dropout_58/StatefulPartitionedCall:output:0dense_55_82378dense_55_82380*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_55_layer_call_and_return_conditional_losses_81664x
IdentityIdentity)dense_55/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_54/StatefulPartitionedCall!^dense_55/StatefulPartitionedCall#^dropout_57/StatefulPartitionedCall#^dropout_58/StatefulPartitionedCall8^token_and_position_embedding_13/StatefulPartitionedCall-^transformer_block_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:??????????: : : : : : : : : : : : : : : : : : : : : : 2D
 dense_54/StatefulPartitionedCall dense_54/StatefulPartitionedCall2D
 dense_55/StatefulPartitionedCall dense_55/StatefulPartitionedCall2H
"dropout_57/StatefulPartitionedCall"dropout_57/StatefulPartitionedCall2H
"dropout_58/StatefulPartitionedCall"dropout_58/StatefulPartitionedCall2r
7token_and_position_embedding_13/StatefulPartitionedCall7token_and_position_embedding_13/StatefulPartitionedCall2\
,transformer_block_13/StatefulPartitionedCall,transformer_block_13/StatefulPartitionedCall:R N
(
_output_shapes
:??????????
"
_user_specified_name
input_14
?

?
C__inference_dense_55_layer_call_and_return_conditional_losses_83618

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
C__inference_dense_54_layer_call_and_return_conditional_losses_81640

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
d
E__inference_dropout_57_layer_call_and_return_conditional_losses_81781

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_58_layer_call_and_return_conditional_losses_81651

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
C__inference_dense_54_layer_call_and_return_conditional_losses_83571

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
d
E__inference_dropout_57_layer_call_and_return_conditional_losses_83551

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
H__inference_sequential_13_layer_call_and_return_conditional_losses_81332
dense_52_input"
dense_52_81321:
??
dense_52_81323:	?"
dense_53_81326:
??
dense_53_81328:	?
identity?? dense_52/StatefulPartitionedCall? dense_53/StatefulPartitionedCall?
 dense_52/StatefulPartitionedCallStatefulPartitionedCalldense_52_inputdense_52_81321dense_52_81323*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_52_layer_call_and_return_conditional_losses_81191?
 dense_53/StatefulPartitionedCallStatefulPartitionedCall)dense_52/StatefulPartitionedCall:output:0dense_53_81326dense_53_81328*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_53_layer_call_and_return_conditional_losses_81227~
IdentityIdentity)dense_53/StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:????????????
NoOpNoOp!^dense_52/StatefulPartitionedCall!^dense_53/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : : : 2D
 dense_52/StatefulPartitionedCall dense_52/StatefulPartitionedCall2D
 dense_53/StatefulPartitionedCall dense_53/StatefulPartitionedCall:] Y
-
_output_shapes
:???????????
(
_user_specified_namedense_52_input
?	
d
E__inference_dropout_58_layer_call_and_return_conditional_losses_83598

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
r
V__inference_global_average_pooling1d_13_layer_call_and_return_conditional_losses_83518

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_57_layer_call_and_return_conditional_losses_81627

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
O__inference_transformer_block_13_layer_call_and_return_conditional_losses_83502

inputs[
Cmulti_head_attention_13_query_einsum_einsum_readvariableop_resource:??L
9multi_head_attention_13_query_add_readvariableop_resource:	?Y
Amulti_head_attention_13_key_einsum_einsum_readvariableop_resource:??J
7multi_head_attention_13_key_add_readvariableop_resource:	?[
Cmulti_head_attention_13_value_einsum_einsum_readvariableop_resource:??L
9multi_head_attention_13_value_add_readvariableop_resource:	?f
Nmulti_head_attention_13_attention_output_einsum_einsum_readvariableop_resource:??S
Dmulti_head_attention_13_attention_output_add_readvariableop_resource:	?C
4layer_normalization_26_mul_3_readvariableop_resource:	?A
2layer_normalization_26_add_readvariableop_resource:	?L
8sequential_13_dense_52_tensordot_readvariableop_resource:
??E
6sequential_13_dense_52_biasadd_readvariableop_resource:	?L
8sequential_13_dense_53_tensordot_readvariableop_resource:
??E
6sequential_13_dense_53_biasadd_readvariableop_resource:	?C
4layer_normalization_27_mul_3_readvariableop_resource:	?A
2layer_normalization_27_add_readvariableop_resource:	?
identity??)layer_normalization_26/add/ReadVariableOp?+layer_normalization_26/mul_3/ReadVariableOp?)layer_normalization_27/add/ReadVariableOp?+layer_normalization_27/mul_3/ReadVariableOp?;multi_head_attention_13/attention_output/add/ReadVariableOp?Emulti_head_attention_13/attention_output/einsum/Einsum/ReadVariableOp?.multi_head_attention_13/key/add/ReadVariableOp?8multi_head_attention_13/key/einsum/Einsum/ReadVariableOp?0multi_head_attention_13/query/add/ReadVariableOp?:multi_head_attention_13/query/einsum/Einsum/ReadVariableOp?0multi_head_attention_13/value/add/ReadVariableOp?:multi_head_attention_13/value/einsum/Einsum/ReadVariableOp?-sequential_13/dense_52/BiasAdd/ReadVariableOp?/sequential_13/dense_52/Tensordot/ReadVariableOp?-sequential_13/dense_53/BiasAdd/ReadVariableOp?/sequential_13/dense_53/Tensordot/ReadVariableOp?
:multi_head_attention_13/query/einsum/Einsum/ReadVariableOpReadVariableOpCmulti_head_attention_13_query_einsum_einsum_readvariableop_resource*$
_output_shapes
:??*
dtype0?
+multi_head_attention_13/query/einsum/EinsumEinsuminputsBmulti_head_attention_13/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*1
_output_shapes
:???????????*
equationabc,cde->abde?
0multi_head_attention_13/query/add/ReadVariableOpReadVariableOp9multi_head_attention_13_query_add_readvariableop_resource*
_output_shapes
:	?*
dtype0?
!multi_head_attention_13/query/addAddV24multi_head_attention_13/query/einsum/Einsum:output:08multi_head_attention_13/query/add/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
8multi_head_attention_13/key/einsum/Einsum/ReadVariableOpReadVariableOpAmulti_head_attention_13_key_einsum_einsum_readvariableop_resource*$
_output_shapes
:??*
dtype0?
)multi_head_attention_13/key/einsum/EinsumEinsuminputs@multi_head_attention_13/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*1
_output_shapes
:???????????*
equationabc,cde->abde?
.multi_head_attention_13/key/add/ReadVariableOpReadVariableOp7multi_head_attention_13_key_add_readvariableop_resource*
_output_shapes
:	?*
dtype0?
multi_head_attention_13/key/addAddV22multi_head_attention_13/key/einsum/Einsum:output:06multi_head_attention_13/key/add/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
:multi_head_attention_13/value/einsum/Einsum/ReadVariableOpReadVariableOpCmulti_head_attention_13_value_einsum_einsum_readvariableop_resource*$
_output_shapes
:??*
dtype0?
+multi_head_attention_13/value/einsum/EinsumEinsuminputsBmulti_head_attention_13/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*1
_output_shapes
:???????????*
equationabc,cde->abde?
0multi_head_attention_13/value/add/ReadVariableOpReadVariableOp9multi_head_attention_13_value_add_readvariableop_resource*
_output_shapes
:	?*
dtype0?
!multi_head_attention_13/value/addAddV24multi_head_attention_13/value/einsum/Einsum:output:08multi_head_attention_13/value/add/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????b
multi_head_attention_13/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *?А=?
multi_head_attention_13/MulMul%multi_head_attention_13/query/add:z:0&multi_head_attention_13/Mul/y:output:0*
T0*1
_output_shapes
:????????????
%multi_head_attention_13/einsum/EinsumEinsum#multi_head_attention_13/key/add:z:0multi_head_attention_13/Mul:z:0*
N*
T0*1
_output_shapes
:???????????*
equationaecd,abcd->acbe?
'multi_head_attention_13/softmax/SoftmaxSoftmax.multi_head_attention_13/einsum/Einsum:output:0*
T0*1
_output_shapes
:????????????
'multi_head_attention_13/einsum_1/EinsumEinsum1multi_head_attention_13/softmax/Softmax:softmax:0%multi_head_attention_13/value/add:z:0*
N*
T0*1
_output_shapes
:???????????*
equationacbe,aecd->abcd?
Emulti_head_attention_13/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpNmulti_head_attention_13_attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:??*
dtype0?
6multi_head_attention_13/attention_output/einsum/EinsumEinsum0multi_head_attention_13/einsum_1/Einsum:output:0Mmulti_head_attention_13/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*-
_output_shapes
:???????????*
equationabcd,cde->abe?
;multi_head_attention_13/attention_output/add/ReadVariableOpReadVariableOpDmulti_head_attention_13_attention_output_add_readvariableop_resource*
_output_shapes	
:?*
dtype0?
,multi_head_attention_13/attention_output/addAddV2?multi_head_attention_13/attention_output/einsum/Einsum:output:0Cmulti_head_attention_13/attention_output/add/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????]
dropout_55/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout_55/dropout/MulMul0multi_head_attention_13/attention_output/add:z:0!dropout_55/dropout/Const:output:0*
T0*-
_output_shapes
:???????????x
dropout_55/dropout/ShapeShape0multi_head_attention_13/attention_output/add:z:0*
T0*
_output_shapes
:?
/dropout_55/dropout/random_uniform/RandomUniformRandomUniform!dropout_55/dropout/Shape:output:0*
T0*-
_output_shapes
:???????????*
dtype0f
!dropout_55/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout_55/dropout/GreaterEqualGreaterEqual8dropout_55/dropout/random_uniform/RandomUniform:output:0*dropout_55/dropout/GreaterEqual/y:output:0*
T0*-
_output_shapes
:????????????
dropout_55/dropout/CastCast#dropout_55/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*-
_output_shapes
:????????????
dropout_55/dropout/Mul_1Muldropout_55/dropout/Mul:z:0dropout_55/dropout/Cast:y:0*
T0*-
_output_shapes
:???????????j
addAddV2inputsdropout_55/dropout/Mul_1:z:0*
T0*-
_output_shapes
:???????????S
layer_normalization_26/ShapeShapeadd:z:0*
T0*
_output_shapes
:t
*layer_normalization_26/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,layer_normalization_26/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,layer_normalization_26/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$layer_normalization_26/strided_sliceStridedSlice%layer_normalization_26/Shape:output:03layer_normalization_26/strided_slice/stack:output:05layer_normalization_26/strided_slice/stack_1:output:05layer_normalization_26/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
layer_normalization_26/mul/xConst*
_output_shapes
: *
dtype0*
value	B :?
layer_normalization_26/mulMul%layer_normalization_26/mul/x:output:0-layer_normalization_26/strided_slice:output:0*
T0*
_output_shapes
: v
,layer_normalization_26/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_26/strided_slice_1StridedSlice%layer_normalization_26/Shape:output:05layer_normalization_26/strided_slice_1/stack:output:07layer_normalization_26/strided_slice_1/stack_1:output:07layer_normalization_26/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
layer_normalization_26/mul_1Mullayer_normalization_26/mul:z:0/layer_normalization_26/strided_slice_1:output:0*
T0*
_output_shapes
: v
,layer_normalization_26/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_26/strided_slice_2StridedSlice%layer_normalization_26/Shape:output:05layer_normalization_26/strided_slice_2/stack:output:07layer_normalization_26/strided_slice_2/stack_1:output:07layer_normalization_26/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
layer_normalization_26/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :?
layer_normalization_26/mul_2Mul'layer_normalization_26/mul_2/x:output:0/layer_normalization_26/strided_slice_2:output:0*
T0*
_output_shapes
: h
&layer_normalization_26/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :h
&layer_normalization_26/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
$layer_normalization_26/Reshape/shapePack/layer_normalization_26/Reshape/shape/0:output:0 layer_normalization_26/mul_1:z:0 layer_normalization_26/mul_2:z:0/layer_normalization_26/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
layer_normalization_26/ReshapeReshapeadd:z:0-layer_normalization_26/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????e
"layer_normalization_26/ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :??
 layer_normalization_26/ones/LessLess layer_normalization_26/mul_1:z:0+layer_normalization_26/ones/Less/y:output:0*
T0*
_output_shapes
: z
"layer_normalization_26/ones/packedPack layer_normalization_26/mul_1:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_26/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
layer_normalization_26/onesFill+layer_normalization_26/ones/packed:output:0*layer_normalization_26/ones/Const:output:0*
T0*#
_output_shapes
:?????????f
#layer_normalization_26/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :??
!layer_normalization_26/zeros/LessLess layer_normalization_26/mul_1:z:0,layer_normalization_26/zeros/Less/y:output:0*
T0*
_output_shapes
: {
#layer_normalization_26/zeros/packedPack layer_normalization_26/mul_1:z:0*
N*
T0*
_output_shapes
:g
"layer_normalization_26/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
layer_normalization_26/zerosFill,layer_normalization_26/zeros/packed:output:0+layer_normalization_26/zeros/Const:output:0*
T0*#
_output_shapes
:?????????_
layer_normalization_26/ConstConst*
_output_shapes
: *
dtype0*
valueB a
layer_normalization_26/Const_1Const*
_output_shapes
: *
dtype0*
valueB ?
'layer_normalization_26/FusedBatchNormV3FusedBatchNormV3'layer_normalization_26/Reshape:output:0$layer_normalization_26/ones:output:0%layer_normalization_26/zeros:output:0%layer_normalization_26/Const:output:0'layer_normalization_26/Const_1:output:0*
T0*
U0*p
_output_shapes^
\:??????????:?????????:?????????:?????????:?????????:*
data_formatNCHW?
 layer_normalization_26/Reshape_1Reshape+layer_normalization_26/FusedBatchNormV3:y:0%layer_normalization_26/Shape:output:0*
T0*-
_output_shapes
:????????????
+layer_normalization_26/mul_3/ReadVariableOpReadVariableOp4layer_normalization_26_mul_3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
layer_normalization_26/mul_3Mul)layer_normalization_26/Reshape_1:output:03layer_normalization_26/mul_3/ReadVariableOp:value:0*
T0*-
_output_shapes
:????????????
)layer_normalization_26/add/ReadVariableOpReadVariableOp2layer_normalization_26_add_readvariableop_resource*
_output_shapes	
:?*
dtype0?
layer_normalization_26/addAddV2 layer_normalization_26/mul_3:z:01layer_normalization_26/add/ReadVariableOp:value:0*
T0*-
_output_shapes
:????????????
/sequential_13/dense_52/Tensordot/ReadVariableOpReadVariableOp8sequential_13_dense_52_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype0o
%sequential_13/dense_52/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:v
%sequential_13/dense_52/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       t
&sequential_13/dense_52/Tensordot/ShapeShapelayer_normalization_26/add:z:0*
T0*
_output_shapes
:p
.sequential_13/dense_52/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
)sequential_13/dense_52/Tensordot/GatherV2GatherV2/sequential_13/dense_52/Tensordot/Shape:output:0.sequential_13/dense_52/Tensordot/free:output:07sequential_13/dense_52/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:r
0sequential_13/dense_52/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+sequential_13/dense_52/Tensordot/GatherV2_1GatherV2/sequential_13/dense_52/Tensordot/Shape:output:0.sequential_13/dense_52/Tensordot/axes:output:09sequential_13/dense_52/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
&sequential_13/dense_52/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
%sequential_13/dense_52/Tensordot/ProdProd2sequential_13/dense_52/Tensordot/GatherV2:output:0/sequential_13/dense_52/Tensordot/Const:output:0*
T0*
_output_shapes
: r
(sequential_13/dense_52/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
'sequential_13/dense_52/Tensordot/Prod_1Prod4sequential_13/dense_52/Tensordot/GatherV2_1:output:01sequential_13/dense_52/Tensordot/Const_1:output:0*
T0*
_output_shapes
: n
,sequential_13/dense_52/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
'sequential_13/dense_52/Tensordot/concatConcatV2.sequential_13/dense_52/Tensordot/free:output:0.sequential_13/dense_52/Tensordot/axes:output:05sequential_13/dense_52/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
&sequential_13/dense_52/Tensordot/stackPack.sequential_13/dense_52/Tensordot/Prod:output:00sequential_13/dense_52/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
*sequential_13/dense_52/Tensordot/transpose	Transposelayer_normalization_26/add:z:00sequential_13/dense_52/Tensordot/concat:output:0*
T0*-
_output_shapes
:????????????
(sequential_13/dense_52/Tensordot/ReshapeReshape.sequential_13/dense_52/Tensordot/transpose:y:0/sequential_13/dense_52/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
'sequential_13/dense_52/Tensordot/MatMulMatMul1sequential_13/dense_52/Tensordot/Reshape:output:07sequential_13/dense_52/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
(sequential_13/dense_52/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?p
.sequential_13/dense_52/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
)sequential_13/dense_52/Tensordot/concat_1ConcatV22sequential_13/dense_52/Tensordot/GatherV2:output:01sequential_13/dense_52/Tensordot/Const_2:output:07sequential_13/dense_52/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
 sequential_13/dense_52/TensordotReshape1sequential_13/dense_52/Tensordot/MatMul:product:02sequential_13/dense_52/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:????????????
-sequential_13/dense_52/BiasAdd/ReadVariableOpReadVariableOp6sequential_13_dense_52_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_13/dense_52/BiasAddBiasAdd)sequential_13/dense_52/Tensordot:output:05sequential_13/dense_52/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:????????????
sequential_13/dense_52/ReluRelu'sequential_13/dense_52/BiasAdd:output:0*
T0*-
_output_shapes
:????????????
/sequential_13/dense_53/Tensordot/ReadVariableOpReadVariableOp8sequential_13_dense_53_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype0o
%sequential_13/dense_53/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:v
%sequential_13/dense_53/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
&sequential_13/dense_53/Tensordot/ShapeShape)sequential_13/dense_52/Relu:activations:0*
T0*
_output_shapes
:p
.sequential_13/dense_53/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
)sequential_13/dense_53/Tensordot/GatherV2GatherV2/sequential_13/dense_53/Tensordot/Shape:output:0.sequential_13/dense_53/Tensordot/free:output:07sequential_13/dense_53/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:r
0sequential_13/dense_53/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+sequential_13/dense_53/Tensordot/GatherV2_1GatherV2/sequential_13/dense_53/Tensordot/Shape:output:0.sequential_13/dense_53/Tensordot/axes:output:09sequential_13/dense_53/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
&sequential_13/dense_53/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
%sequential_13/dense_53/Tensordot/ProdProd2sequential_13/dense_53/Tensordot/GatherV2:output:0/sequential_13/dense_53/Tensordot/Const:output:0*
T0*
_output_shapes
: r
(sequential_13/dense_53/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
'sequential_13/dense_53/Tensordot/Prod_1Prod4sequential_13/dense_53/Tensordot/GatherV2_1:output:01sequential_13/dense_53/Tensordot/Const_1:output:0*
T0*
_output_shapes
: n
,sequential_13/dense_53/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
'sequential_13/dense_53/Tensordot/concatConcatV2.sequential_13/dense_53/Tensordot/free:output:0.sequential_13/dense_53/Tensordot/axes:output:05sequential_13/dense_53/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
&sequential_13/dense_53/Tensordot/stackPack.sequential_13/dense_53/Tensordot/Prod:output:00sequential_13/dense_53/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
*sequential_13/dense_53/Tensordot/transpose	Transpose)sequential_13/dense_52/Relu:activations:00sequential_13/dense_53/Tensordot/concat:output:0*
T0*-
_output_shapes
:????????????
(sequential_13/dense_53/Tensordot/ReshapeReshape.sequential_13/dense_53/Tensordot/transpose:y:0/sequential_13/dense_53/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
'sequential_13/dense_53/Tensordot/MatMulMatMul1sequential_13/dense_53/Tensordot/Reshape:output:07sequential_13/dense_53/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
(sequential_13/dense_53/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?p
.sequential_13/dense_53/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
)sequential_13/dense_53/Tensordot/concat_1ConcatV22sequential_13/dense_53/Tensordot/GatherV2:output:01sequential_13/dense_53/Tensordot/Const_2:output:07sequential_13/dense_53/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
 sequential_13/dense_53/TensordotReshape1sequential_13/dense_53/Tensordot/MatMul:product:02sequential_13/dense_53/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:????????????
-sequential_13/dense_53/BiasAdd/ReadVariableOpReadVariableOp6sequential_13_dense_53_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_13/dense_53/BiasAddBiasAdd)sequential_13/dense_53/Tensordot:output:05sequential_13/dense_53/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????]
dropout_56/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout_56/dropout/MulMul'sequential_13/dense_53/BiasAdd:output:0!dropout_56/dropout/Const:output:0*
T0*-
_output_shapes
:???????????o
dropout_56/dropout/ShapeShape'sequential_13/dense_53/BiasAdd:output:0*
T0*
_output_shapes
:?
/dropout_56/dropout/random_uniform/RandomUniformRandomUniform!dropout_56/dropout/Shape:output:0*
T0*-
_output_shapes
:???????????*
dtype0f
!dropout_56/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout_56/dropout/GreaterEqualGreaterEqual8dropout_56/dropout/random_uniform/RandomUniform:output:0*dropout_56/dropout/GreaterEqual/y:output:0*
T0*-
_output_shapes
:????????????
dropout_56/dropout/CastCast#dropout_56/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*-
_output_shapes
:????????????
dropout_56/dropout/Mul_1Muldropout_56/dropout/Mul:z:0dropout_56/dropout/Cast:y:0*
T0*-
_output_shapes
:????????????
add_1AddV2layer_normalization_26/add:z:0dropout_56/dropout/Mul_1:z:0*
T0*-
_output_shapes
:???????????U
layer_normalization_27/ShapeShape	add_1:z:0*
T0*
_output_shapes
:t
*layer_normalization_27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,layer_normalization_27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,layer_normalization_27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$layer_normalization_27/strided_sliceStridedSlice%layer_normalization_27/Shape:output:03layer_normalization_27/strided_slice/stack:output:05layer_normalization_27/strided_slice/stack_1:output:05layer_normalization_27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
layer_normalization_27/mul/xConst*
_output_shapes
: *
dtype0*
value	B :?
layer_normalization_27/mulMul%layer_normalization_27/mul/x:output:0-layer_normalization_27/strided_slice:output:0*
T0*
_output_shapes
: v
,layer_normalization_27/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_27/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_27/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_27/strided_slice_1StridedSlice%layer_normalization_27/Shape:output:05layer_normalization_27/strided_slice_1/stack:output:07layer_normalization_27/strided_slice_1/stack_1:output:07layer_normalization_27/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
layer_normalization_27/mul_1Mullayer_normalization_27/mul:z:0/layer_normalization_27/strided_slice_1:output:0*
T0*
_output_shapes
: v
,layer_normalization_27/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_27/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_27/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_27/strided_slice_2StridedSlice%layer_normalization_27/Shape:output:05layer_normalization_27/strided_slice_2/stack:output:07layer_normalization_27/strided_slice_2/stack_1:output:07layer_normalization_27/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
layer_normalization_27/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :?
layer_normalization_27/mul_2Mul'layer_normalization_27/mul_2/x:output:0/layer_normalization_27/strided_slice_2:output:0*
T0*
_output_shapes
: h
&layer_normalization_27/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :h
&layer_normalization_27/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
$layer_normalization_27/Reshape/shapePack/layer_normalization_27/Reshape/shape/0:output:0 layer_normalization_27/mul_1:z:0 layer_normalization_27/mul_2:z:0/layer_normalization_27/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
layer_normalization_27/ReshapeReshape	add_1:z:0-layer_normalization_27/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????e
"layer_normalization_27/ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :??
 layer_normalization_27/ones/LessLess layer_normalization_27/mul_1:z:0+layer_normalization_27/ones/Less/y:output:0*
T0*
_output_shapes
: z
"layer_normalization_27/ones/packedPack layer_normalization_27/mul_1:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_27/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
layer_normalization_27/onesFill+layer_normalization_27/ones/packed:output:0*layer_normalization_27/ones/Const:output:0*
T0*#
_output_shapes
:?????????f
#layer_normalization_27/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :??
!layer_normalization_27/zeros/LessLess layer_normalization_27/mul_1:z:0,layer_normalization_27/zeros/Less/y:output:0*
T0*
_output_shapes
: {
#layer_normalization_27/zeros/packedPack layer_normalization_27/mul_1:z:0*
N*
T0*
_output_shapes
:g
"layer_normalization_27/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
layer_normalization_27/zerosFill,layer_normalization_27/zeros/packed:output:0+layer_normalization_27/zeros/Const:output:0*
T0*#
_output_shapes
:?????????_
layer_normalization_27/ConstConst*
_output_shapes
: *
dtype0*
valueB a
layer_normalization_27/Const_1Const*
_output_shapes
: *
dtype0*
valueB ?
'layer_normalization_27/FusedBatchNormV3FusedBatchNormV3'layer_normalization_27/Reshape:output:0$layer_normalization_27/ones:output:0%layer_normalization_27/zeros:output:0%layer_normalization_27/Const:output:0'layer_normalization_27/Const_1:output:0*
T0*
U0*p
_output_shapes^
\:??????????:?????????:?????????:?????????:?????????:*
data_formatNCHW?
 layer_normalization_27/Reshape_1Reshape+layer_normalization_27/FusedBatchNormV3:y:0%layer_normalization_27/Shape:output:0*
T0*-
_output_shapes
:????????????
+layer_normalization_27/mul_3/ReadVariableOpReadVariableOp4layer_normalization_27_mul_3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
layer_normalization_27/mul_3Mul)layer_normalization_27/Reshape_1:output:03layer_normalization_27/mul_3/ReadVariableOp:value:0*
T0*-
_output_shapes
:????????????
)layer_normalization_27/add/ReadVariableOpReadVariableOp2layer_normalization_27_add_readvariableop_resource*
_output_shapes	
:?*
dtype0?
layer_normalization_27/addAddV2 layer_normalization_27/mul_3:z:01layer_normalization_27/add/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????s
IdentityIdentitylayer_normalization_27/add:z:0^NoOp*
T0*-
_output_shapes
:????????????
NoOpNoOp*^layer_normalization_26/add/ReadVariableOp,^layer_normalization_26/mul_3/ReadVariableOp*^layer_normalization_27/add/ReadVariableOp,^layer_normalization_27/mul_3/ReadVariableOp<^multi_head_attention_13/attention_output/add/ReadVariableOpF^multi_head_attention_13/attention_output/einsum/Einsum/ReadVariableOp/^multi_head_attention_13/key/add/ReadVariableOp9^multi_head_attention_13/key/einsum/Einsum/ReadVariableOp1^multi_head_attention_13/query/add/ReadVariableOp;^multi_head_attention_13/query/einsum/Einsum/ReadVariableOp1^multi_head_attention_13/value/add/ReadVariableOp;^multi_head_attention_13/value/einsum/Einsum/ReadVariableOp.^sequential_13/dense_52/BiasAdd/ReadVariableOp0^sequential_13/dense_52/Tensordot/ReadVariableOp.^sequential_13/dense_53/BiasAdd/ReadVariableOp0^sequential_13/dense_53/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:???????????: : : : : : : : : : : : : : : : 2V
)layer_normalization_26/add/ReadVariableOp)layer_normalization_26/add/ReadVariableOp2Z
+layer_normalization_26/mul_3/ReadVariableOp+layer_normalization_26/mul_3/ReadVariableOp2V
)layer_normalization_27/add/ReadVariableOp)layer_normalization_27/add/ReadVariableOp2Z
+layer_normalization_27/mul_3/ReadVariableOp+layer_normalization_27/mul_3/ReadVariableOp2z
;multi_head_attention_13/attention_output/add/ReadVariableOp;multi_head_attention_13/attention_output/add/ReadVariableOp2?
Emulti_head_attention_13/attention_output/einsum/Einsum/ReadVariableOpEmulti_head_attention_13/attention_output/einsum/Einsum/ReadVariableOp2`
.multi_head_attention_13/key/add/ReadVariableOp.multi_head_attention_13/key/add/ReadVariableOp2t
8multi_head_attention_13/key/einsum/Einsum/ReadVariableOp8multi_head_attention_13/key/einsum/Einsum/ReadVariableOp2d
0multi_head_attention_13/query/add/ReadVariableOp0multi_head_attention_13/query/add/ReadVariableOp2x
:multi_head_attention_13/query/einsum/Einsum/ReadVariableOp:multi_head_attention_13/query/einsum/Einsum/ReadVariableOp2d
0multi_head_attention_13/value/add/ReadVariableOp0multi_head_attention_13/value/add/ReadVariableOp2x
:multi_head_attention_13/value/einsum/Einsum/ReadVariableOp:multi_head_attention_13/value/einsum/Einsum/ReadVariableOp2^
-sequential_13/dense_52/BiasAdd/ReadVariableOp-sequential_13/dense_52/BiasAdd/ReadVariableOp2b
/sequential_13/dense_52/Tensordot/ReadVariableOp/sequential_13/dense_52/Tensordot/ReadVariableOp2^
-sequential_13/dense_53/BiasAdd/ReadVariableOp-sequential_13/dense_53/BiasAdd/ReadVariableOp2b
/sequential_13/dense_53/Tensordot/ReadVariableOp/sequential_13/dense_53/Tensordot/ReadVariableOp:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
(__inference_model_13_layer_call_fn_81718
input_14
unknown:
??
	unknown_0:
?)?!
	unknown_1:??
	unknown_2:	?!
	unknown_3:??
	unknown_4:	?!
	unknown_5:??
	unknown_6:	?!
	unknown_7:??
	unknown_8:	?
	unknown_9:	?

unknown_10:	?

unknown_11:
??

unknown_12:	?

unknown_13:
??

unknown_14:	?

unknown_15:	?

unknown_16:	?

unknown_17:	?

unknown_18:

unknown_19:

unknown_20:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_14unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_model_13_layer_call_and_return_conditional_losses_81671o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:??????????: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:??????????
"
_user_specified_name
input_14
??
?<
!__inference__traced_restore_84320
file_prefix3
 assignvariableop_dense_54_kernel:	?.
 assignvariableop_1_dense_54_bias:4
"assignvariableop_2_dense_55_kernel:.
 assignvariableop_3_dense_55_bias:&
assignvariableop_4_adam_iter:	 (
assignvariableop_5_adam_beta_1: (
assignvariableop_6_adam_beta_2: '
assignvariableop_7_adam_decay: /
%assignvariableop_8_adam_learning_rate: ^
Jassignvariableop_9_token_and_position_embedding_13_embedding_26_embeddings:
?)?_
Kassignvariableop_10_token_and_position_embedding_13_embedding_27_embeddings:
??e
Massignvariableop_11_transformer_block_13_multi_head_attention_13_query_kernel:??^
Kassignvariableop_12_transformer_block_13_multi_head_attention_13_query_bias:	?c
Kassignvariableop_13_transformer_block_13_multi_head_attention_13_key_kernel:??\
Iassignvariableop_14_transformer_block_13_multi_head_attention_13_key_bias:	?e
Massignvariableop_15_transformer_block_13_multi_head_attention_13_value_kernel:??^
Kassignvariableop_16_transformer_block_13_multi_head_attention_13_value_bias:	?p
Xassignvariableop_17_transformer_block_13_multi_head_attention_13_attention_output_kernel:??e
Vassignvariableop_18_transformer_block_13_multi_head_attention_13_attention_output_bias:	?7
#assignvariableop_19_dense_52_kernel:
??0
!assignvariableop_20_dense_52_bias:	?7
#assignvariableop_21_dense_53_kernel:
??0
!assignvariableop_22_dense_53_bias:	?T
Eassignvariableop_23_transformer_block_13_layer_normalization_26_gamma:	?S
Dassignvariableop_24_transformer_block_13_layer_normalization_26_beta:	?T
Eassignvariableop_25_transformer_block_13_layer_normalization_27_gamma:	?S
Dassignvariableop_26_transformer_block_13_layer_normalization_27_beta:	?#
assignvariableop_27_total: #
assignvariableop_28_count: %
assignvariableop_29_total_1: %
assignvariableop_30_count_1: =
*assignvariableop_31_adam_dense_54_kernel_m:	?6
(assignvariableop_32_adam_dense_54_bias_m:<
*assignvariableop_33_adam_dense_55_kernel_m:6
(assignvariableop_34_adam_dense_55_bias_m:f
Rassignvariableop_35_adam_token_and_position_embedding_13_embedding_26_embeddings_m:
?)?f
Rassignvariableop_36_adam_token_and_position_embedding_13_embedding_27_embeddings_m:
??l
Tassignvariableop_37_adam_transformer_block_13_multi_head_attention_13_query_kernel_m:??e
Rassignvariableop_38_adam_transformer_block_13_multi_head_attention_13_query_bias_m:	?j
Rassignvariableop_39_adam_transformer_block_13_multi_head_attention_13_key_kernel_m:??c
Passignvariableop_40_adam_transformer_block_13_multi_head_attention_13_key_bias_m:	?l
Tassignvariableop_41_adam_transformer_block_13_multi_head_attention_13_value_kernel_m:??e
Rassignvariableop_42_adam_transformer_block_13_multi_head_attention_13_value_bias_m:	?w
_assignvariableop_43_adam_transformer_block_13_multi_head_attention_13_attention_output_kernel_m:??l
]assignvariableop_44_adam_transformer_block_13_multi_head_attention_13_attention_output_bias_m:	?>
*assignvariableop_45_adam_dense_52_kernel_m:
??7
(assignvariableop_46_adam_dense_52_bias_m:	?>
*assignvariableop_47_adam_dense_53_kernel_m:
??7
(assignvariableop_48_adam_dense_53_bias_m:	?[
Lassignvariableop_49_adam_transformer_block_13_layer_normalization_26_gamma_m:	?Z
Kassignvariableop_50_adam_transformer_block_13_layer_normalization_26_beta_m:	?[
Lassignvariableop_51_adam_transformer_block_13_layer_normalization_27_gamma_m:	?Z
Kassignvariableop_52_adam_transformer_block_13_layer_normalization_27_beta_m:	?=
*assignvariableop_53_adam_dense_54_kernel_v:	?6
(assignvariableop_54_adam_dense_54_bias_v:<
*assignvariableop_55_adam_dense_55_kernel_v:6
(assignvariableop_56_adam_dense_55_bias_v:f
Rassignvariableop_57_adam_token_and_position_embedding_13_embedding_26_embeddings_v:
?)?f
Rassignvariableop_58_adam_token_and_position_embedding_13_embedding_27_embeddings_v:
??l
Tassignvariableop_59_adam_transformer_block_13_multi_head_attention_13_query_kernel_v:??e
Rassignvariableop_60_adam_transformer_block_13_multi_head_attention_13_query_bias_v:	?j
Rassignvariableop_61_adam_transformer_block_13_multi_head_attention_13_key_kernel_v:??c
Passignvariableop_62_adam_transformer_block_13_multi_head_attention_13_key_bias_v:	?l
Tassignvariableop_63_adam_transformer_block_13_multi_head_attention_13_value_kernel_v:??e
Rassignvariableop_64_adam_transformer_block_13_multi_head_attention_13_value_bias_v:	?w
_assignvariableop_65_adam_transformer_block_13_multi_head_attention_13_attention_output_kernel_v:??l
]assignvariableop_66_adam_transformer_block_13_multi_head_attention_13_attention_output_bias_v:	?>
*assignvariableop_67_adam_dense_52_kernel_v:
??7
(assignvariableop_68_adam_dense_52_bias_v:	?>
*assignvariableop_69_adam_dense_53_kernel_v:
??7
(assignvariableop_70_adam_dense_53_bias_v:	?[
Lassignvariableop_71_adam_transformer_block_13_layer_normalization_26_gamma_v:	?Z
Kassignvariableop_72_adam_transformer_block_13_layer_normalization_26_beta_v:	?[
Lassignvariableop_73_adam_transformer_block_13_layer_normalization_27_gamma_v:	?Z
Kassignvariableop_74_adam_transformer_block_13_layer_normalization_27_beta_v:	?
identity_76??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_8?AssignVariableOp_9?$
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*?#
value?#B?#LB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*?
value?B?LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Z
dtypesP
N2L	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp assignvariableop_dense_54_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_54_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_55_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_55_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpJassignvariableop_9_token_and_position_embedding_13_embedding_26_embeddingsIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpKassignvariableop_10_token_and_position_embedding_13_embedding_27_embeddingsIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpMassignvariableop_11_transformer_block_13_multi_head_attention_13_query_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpKassignvariableop_12_transformer_block_13_multi_head_attention_13_query_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpKassignvariableop_13_transformer_block_13_multi_head_attention_13_key_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpIassignvariableop_14_transformer_block_13_multi_head_attention_13_key_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpMassignvariableop_15_transformer_block_13_multi_head_attention_13_value_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOpKassignvariableop_16_transformer_block_13_multi_head_attention_13_value_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOpXassignvariableop_17_transformer_block_13_multi_head_attention_13_attention_output_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpVassignvariableop_18_transformer_block_13_multi_head_attention_13_attention_output_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp#assignvariableop_19_dense_52_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp!assignvariableop_20_dense_52_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp#assignvariableop_21_dense_53_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp!assignvariableop_22_dense_53_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOpEassignvariableop_23_transformer_block_13_layer_normalization_26_gammaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOpDassignvariableop_24_transformer_block_13_layer_normalization_26_betaIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOpEassignvariableop_25_transformer_block_13_layer_normalization_27_gammaIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOpDassignvariableop_26_transformer_block_13_layer_normalization_27_betaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOpassignvariableop_27_totalIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOpassignvariableop_28_countIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOpassignvariableop_29_total_1Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOpassignvariableop_30_count_1Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_dense_54_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_dense_54_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_dense_55_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_dense_55_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOpRassignvariableop_35_adam_token_and_position_embedding_13_embedding_26_embeddings_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOpRassignvariableop_36_adam_token_and_position_embedding_13_embedding_27_embeddings_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOpTassignvariableop_37_adam_transformer_block_13_multi_head_attention_13_query_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOpRassignvariableop_38_adam_transformer_block_13_multi_head_attention_13_query_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOpRassignvariableop_39_adam_transformer_block_13_multi_head_attention_13_key_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOpPassignvariableop_40_adam_transformer_block_13_multi_head_attention_13_key_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOpTassignvariableop_41_adam_transformer_block_13_multi_head_attention_13_value_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOpRassignvariableop_42_adam_transformer_block_13_multi_head_attention_13_value_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp_assignvariableop_43_adam_transformer_block_13_multi_head_attention_13_attention_output_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp]assignvariableop_44_adam_transformer_block_13_multi_head_attention_13_attention_output_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_dense_52_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_dense_52_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_dense_53_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_dense_53_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOpLassignvariableop_49_adam_transformer_block_13_layer_normalization_26_gamma_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_50AssignVariableOpKassignvariableop_50_adam_transformer_block_13_layer_normalization_26_beta_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_51AssignVariableOpLassignvariableop_51_adam_transformer_block_13_layer_normalization_27_gamma_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_52AssignVariableOpKassignvariableop_52_adam_transformer_block_13_layer_normalization_27_beta_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_53AssignVariableOp*assignvariableop_53_adam_dense_54_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_54AssignVariableOp(assignvariableop_54_adam_dense_54_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_dense_55_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_56AssignVariableOp(assignvariableop_56_adam_dense_55_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_57AssignVariableOpRassignvariableop_57_adam_token_and_position_embedding_13_embedding_26_embeddings_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_58AssignVariableOpRassignvariableop_58_adam_token_and_position_embedding_13_embedding_27_embeddings_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_59AssignVariableOpTassignvariableop_59_adam_transformer_block_13_multi_head_attention_13_query_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_60AssignVariableOpRassignvariableop_60_adam_transformer_block_13_multi_head_attention_13_query_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_61AssignVariableOpRassignvariableop_61_adam_transformer_block_13_multi_head_attention_13_key_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_62AssignVariableOpPassignvariableop_62_adam_transformer_block_13_multi_head_attention_13_key_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_63AssignVariableOpTassignvariableop_63_adam_transformer_block_13_multi_head_attention_13_value_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_64AssignVariableOpRassignvariableop_64_adam_transformer_block_13_multi_head_attention_13_value_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_65AssignVariableOp_assignvariableop_65_adam_transformer_block_13_multi_head_attention_13_attention_output_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_66AssignVariableOp]assignvariableop_66_adam_transformer_block_13_multi_head_attention_13_attention_output_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_67AssignVariableOp*assignvariableop_67_adam_dense_52_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_68AssignVariableOp(assignvariableop_68_adam_dense_52_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_69AssignVariableOp*assignvariableop_69_adam_dense_53_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_70AssignVariableOp(assignvariableop_70_adam_dense_53_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_71AssignVariableOpLassignvariableop_71_adam_transformer_block_13_layer_normalization_26_gamma_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_72AssignVariableOpKassignvariableop_72_adam_transformer_block_13_layer_normalization_26_beta_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_73AssignVariableOpLassignvariableop_73_adam_transformer_block_13_layer_normalization_27_gamma_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_74AssignVariableOpKassignvariableop_74_adam_transformer_block_13_layer_normalization_27_beta_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_75Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_76IdentityIdentity_75:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_76Identity_76:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
C__inference_dense_52_layer_call_and_return_conditional_losses_83798

inputs5
!tensordot_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:{
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*-
_output_shapes
:????????????
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*-
_output_shapes
:???????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????V
ReluReluBiasAdd:output:0*
T0*-
_output_shapes
:???????????g
IdentityIdentityRelu:activations:0^NoOp*
T0*-
_output_shapes
:???????????z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?
O__inference_transformer_block_13_layer_call_and_return_conditional_losses_81581

inputs[
Cmulti_head_attention_13_query_einsum_einsum_readvariableop_resource:??L
9multi_head_attention_13_query_add_readvariableop_resource:	?Y
Amulti_head_attention_13_key_einsum_einsum_readvariableop_resource:??J
7multi_head_attention_13_key_add_readvariableop_resource:	?[
Cmulti_head_attention_13_value_einsum_einsum_readvariableop_resource:??L
9multi_head_attention_13_value_add_readvariableop_resource:	?f
Nmulti_head_attention_13_attention_output_einsum_einsum_readvariableop_resource:??S
Dmulti_head_attention_13_attention_output_add_readvariableop_resource:	?C
4layer_normalization_26_mul_3_readvariableop_resource:	?A
2layer_normalization_26_add_readvariableop_resource:	?L
8sequential_13_dense_52_tensordot_readvariableop_resource:
??E
6sequential_13_dense_52_biasadd_readvariableop_resource:	?L
8sequential_13_dense_53_tensordot_readvariableop_resource:
??E
6sequential_13_dense_53_biasadd_readvariableop_resource:	?C
4layer_normalization_27_mul_3_readvariableop_resource:	?A
2layer_normalization_27_add_readvariableop_resource:	?
identity??)layer_normalization_26/add/ReadVariableOp?+layer_normalization_26/mul_3/ReadVariableOp?)layer_normalization_27/add/ReadVariableOp?+layer_normalization_27/mul_3/ReadVariableOp?;multi_head_attention_13/attention_output/add/ReadVariableOp?Emulti_head_attention_13/attention_output/einsum/Einsum/ReadVariableOp?.multi_head_attention_13/key/add/ReadVariableOp?8multi_head_attention_13/key/einsum/Einsum/ReadVariableOp?0multi_head_attention_13/query/add/ReadVariableOp?:multi_head_attention_13/query/einsum/Einsum/ReadVariableOp?0multi_head_attention_13/value/add/ReadVariableOp?:multi_head_attention_13/value/einsum/Einsum/ReadVariableOp?-sequential_13/dense_52/BiasAdd/ReadVariableOp?/sequential_13/dense_52/Tensordot/ReadVariableOp?-sequential_13/dense_53/BiasAdd/ReadVariableOp?/sequential_13/dense_53/Tensordot/ReadVariableOp?
:multi_head_attention_13/query/einsum/Einsum/ReadVariableOpReadVariableOpCmulti_head_attention_13_query_einsum_einsum_readvariableop_resource*$
_output_shapes
:??*
dtype0?
+multi_head_attention_13/query/einsum/EinsumEinsuminputsBmulti_head_attention_13/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*1
_output_shapes
:???????????*
equationabc,cde->abde?
0multi_head_attention_13/query/add/ReadVariableOpReadVariableOp9multi_head_attention_13_query_add_readvariableop_resource*
_output_shapes
:	?*
dtype0?
!multi_head_attention_13/query/addAddV24multi_head_attention_13/query/einsum/Einsum:output:08multi_head_attention_13/query/add/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
8multi_head_attention_13/key/einsum/Einsum/ReadVariableOpReadVariableOpAmulti_head_attention_13_key_einsum_einsum_readvariableop_resource*$
_output_shapes
:??*
dtype0?
)multi_head_attention_13/key/einsum/EinsumEinsuminputs@multi_head_attention_13/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*1
_output_shapes
:???????????*
equationabc,cde->abde?
.multi_head_attention_13/key/add/ReadVariableOpReadVariableOp7multi_head_attention_13_key_add_readvariableop_resource*
_output_shapes
:	?*
dtype0?
multi_head_attention_13/key/addAddV22multi_head_attention_13/key/einsum/Einsum:output:06multi_head_attention_13/key/add/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
:multi_head_attention_13/value/einsum/Einsum/ReadVariableOpReadVariableOpCmulti_head_attention_13_value_einsum_einsum_readvariableop_resource*$
_output_shapes
:??*
dtype0?
+multi_head_attention_13/value/einsum/EinsumEinsuminputsBmulti_head_attention_13/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*1
_output_shapes
:???????????*
equationabc,cde->abde?
0multi_head_attention_13/value/add/ReadVariableOpReadVariableOp9multi_head_attention_13_value_add_readvariableop_resource*
_output_shapes
:	?*
dtype0?
!multi_head_attention_13/value/addAddV24multi_head_attention_13/value/einsum/Einsum:output:08multi_head_attention_13/value/add/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????b
multi_head_attention_13/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *?А=?
multi_head_attention_13/MulMul%multi_head_attention_13/query/add:z:0&multi_head_attention_13/Mul/y:output:0*
T0*1
_output_shapes
:????????????
%multi_head_attention_13/einsum/EinsumEinsum#multi_head_attention_13/key/add:z:0multi_head_attention_13/Mul:z:0*
N*
T0*1
_output_shapes
:???????????*
equationaecd,abcd->acbe?
'multi_head_attention_13/softmax/SoftmaxSoftmax.multi_head_attention_13/einsum/Einsum:output:0*
T0*1
_output_shapes
:????????????
(multi_head_attention_13/dropout/IdentityIdentity1multi_head_attention_13/softmax/Softmax:softmax:0*
T0*1
_output_shapes
:????????????
'multi_head_attention_13/einsum_1/EinsumEinsum1multi_head_attention_13/dropout/Identity:output:0%multi_head_attention_13/value/add:z:0*
N*
T0*1
_output_shapes
:???????????*
equationacbe,aecd->abcd?
Emulti_head_attention_13/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpNmulti_head_attention_13_attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:??*
dtype0?
6multi_head_attention_13/attention_output/einsum/EinsumEinsum0multi_head_attention_13/einsum_1/Einsum:output:0Mmulti_head_attention_13/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*-
_output_shapes
:???????????*
equationabcd,cde->abe?
;multi_head_attention_13/attention_output/add/ReadVariableOpReadVariableOpDmulti_head_attention_13_attention_output_add_readvariableop_resource*
_output_shapes	
:?*
dtype0?
,multi_head_attention_13/attention_output/addAddV2?multi_head_attention_13/attention_output/einsum/Einsum:output:0Cmulti_head_attention_13/attention_output/add/ReadVariableOp:value:0*
T0*-
_output_shapes
:????????????
dropout_55/IdentityIdentity0multi_head_attention_13/attention_output/add:z:0*
T0*-
_output_shapes
:???????????j
addAddV2inputsdropout_55/Identity:output:0*
T0*-
_output_shapes
:???????????S
layer_normalization_26/ShapeShapeadd:z:0*
T0*
_output_shapes
:t
*layer_normalization_26/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,layer_normalization_26/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,layer_normalization_26/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$layer_normalization_26/strided_sliceStridedSlice%layer_normalization_26/Shape:output:03layer_normalization_26/strided_slice/stack:output:05layer_normalization_26/strided_slice/stack_1:output:05layer_normalization_26/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
layer_normalization_26/mul/xConst*
_output_shapes
: *
dtype0*
value	B :?
layer_normalization_26/mulMul%layer_normalization_26/mul/x:output:0-layer_normalization_26/strided_slice:output:0*
T0*
_output_shapes
: v
,layer_normalization_26/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_26/strided_slice_1StridedSlice%layer_normalization_26/Shape:output:05layer_normalization_26/strided_slice_1/stack:output:07layer_normalization_26/strided_slice_1/stack_1:output:07layer_normalization_26/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
layer_normalization_26/mul_1Mullayer_normalization_26/mul:z:0/layer_normalization_26/strided_slice_1:output:0*
T0*
_output_shapes
: v
,layer_normalization_26/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_26/strided_slice_2StridedSlice%layer_normalization_26/Shape:output:05layer_normalization_26/strided_slice_2/stack:output:07layer_normalization_26/strided_slice_2/stack_1:output:07layer_normalization_26/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
layer_normalization_26/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :?
layer_normalization_26/mul_2Mul'layer_normalization_26/mul_2/x:output:0/layer_normalization_26/strided_slice_2:output:0*
T0*
_output_shapes
: h
&layer_normalization_26/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :h
&layer_normalization_26/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
$layer_normalization_26/Reshape/shapePack/layer_normalization_26/Reshape/shape/0:output:0 layer_normalization_26/mul_1:z:0 layer_normalization_26/mul_2:z:0/layer_normalization_26/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
layer_normalization_26/ReshapeReshapeadd:z:0-layer_normalization_26/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????e
"layer_normalization_26/ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :??
 layer_normalization_26/ones/LessLess layer_normalization_26/mul_1:z:0+layer_normalization_26/ones/Less/y:output:0*
T0*
_output_shapes
: z
"layer_normalization_26/ones/packedPack layer_normalization_26/mul_1:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_26/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
layer_normalization_26/onesFill+layer_normalization_26/ones/packed:output:0*layer_normalization_26/ones/Const:output:0*
T0*#
_output_shapes
:?????????f
#layer_normalization_26/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :??
!layer_normalization_26/zeros/LessLess layer_normalization_26/mul_1:z:0,layer_normalization_26/zeros/Less/y:output:0*
T0*
_output_shapes
: {
#layer_normalization_26/zeros/packedPack layer_normalization_26/mul_1:z:0*
N*
T0*
_output_shapes
:g
"layer_normalization_26/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
layer_normalization_26/zerosFill,layer_normalization_26/zeros/packed:output:0+layer_normalization_26/zeros/Const:output:0*
T0*#
_output_shapes
:?????????_
layer_normalization_26/ConstConst*
_output_shapes
: *
dtype0*
valueB a
layer_normalization_26/Const_1Const*
_output_shapes
: *
dtype0*
valueB ?
'layer_normalization_26/FusedBatchNormV3FusedBatchNormV3'layer_normalization_26/Reshape:output:0$layer_normalization_26/ones:output:0%layer_normalization_26/zeros:output:0%layer_normalization_26/Const:output:0'layer_normalization_26/Const_1:output:0*
T0*
U0*p
_output_shapes^
\:??????????:?????????:?????????:?????????:?????????:*
data_formatNCHW?
 layer_normalization_26/Reshape_1Reshape+layer_normalization_26/FusedBatchNormV3:y:0%layer_normalization_26/Shape:output:0*
T0*-
_output_shapes
:????????????
+layer_normalization_26/mul_3/ReadVariableOpReadVariableOp4layer_normalization_26_mul_3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
layer_normalization_26/mul_3Mul)layer_normalization_26/Reshape_1:output:03layer_normalization_26/mul_3/ReadVariableOp:value:0*
T0*-
_output_shapes
:????????????
)layer_normalization_26/add/ReadVariableOpReadVariableOp2layer_normalization_26_add_readvariableop_resource*
_output_shapes	
:?*
dtype0?
layer_normalization_26/addAddV2 layer_normalization_26/mul_3:z:01layer_normalization_26/add/ReadVariableOp:value:0*
T0*-
_output_shapes
:????????????
/sequential_13/dense_52/Tensordot/ReadVariableOpReadVariableOp8sequential_13_dense_52_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype0o
%sequential_13/dense_52/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:v
%sequential_13/dense_52/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       t
&sequential_13/dense_52/Tensordot/ShapeShapelayer_normalization_26/add:z:0*
T0*
_output_shapes
:p
.sequential_13/dense_52/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
)sequential_13/dense_52/Tensordot/GatherV2GatherV2/sequential_13/dense_52/Tensordot/Shape:output:0.sequential_13/dense_52/Tensordot/free:output:07sequential_13/dense_52/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:r
0sequential_13/dense_52/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+sequential_13/dense_52/Tensordot/GatherV2_1GatherV2/sequential_13/dense_52/Tensordot/Shape:output:0.sequential_13/dense_52/Tensordot/axes:output:09sequential_13/dense_52/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
&sequential_13/dense_52/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
%sequential_13/dense_52/Tensordot/ProdProd2sequential_13/dense_52/Tensordot/GatherV2:output:0/sequential_13/dense_52/Tensordot/Const:output:0*
T0*
_output_shapes
: r
(sequential_13/dense_52/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
'sequential_13/dense_52/Tensordot/Prod_1Prod4sequential_13/dense_52/Tensordot/GatherV2_1:output:01sequential_13/dense_52/Tensordot/Const_1:output:0*
T0*
_output_shapes
: n
,sequential_13/dense_52/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
'sequential_13/dense_52/Tensordot/concatConcatV2.sequential_13/dense_52/Tensordot/free:output:0.sequential_13/dense_52/Tensordot/axes:output:05sequential_13/dense_52/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
&sequential_13/dense_52/Tensordot/stackPack.sequential_13/dense_52/Tensordot/Prod:output:00sequential_13/dense_52/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
*sequential_13/dense_52/Tensordot/transpose	Transposelayer_normalization_26/add:z:00sequential_13/dense_52/Tensordot/concat:output:0*
T0*-
_output_shapes
:????????????
(sequential_13/dense_52/Tensordot/ReshapeReshape.sequential_13/dense_52/Tensordot/transpose:y:0/sequential_13/dense_52/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
'sequential_13/dense_52/Tensordot/MatMulMatMul1sequential_13/dense_52/Tensordot/Reshape:output:07sequential_13/dense_52/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
(sequential_13/dense_52/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?p
.sequential_13/dense_52/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
)sequential_13/dense_52/Tensordot/concat_1ConcatV22sequential_13/dense_52/Tensordot/GatherV2:output:01sequential_13/dense_52/Tensordot/Const_2:output:07sequential_13/dense_52/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
 sequential_13/dense_52/TensordotReshape1sequential_13/dense_52/Tensordot/MatMul:product:02sequential_13/dense_52/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:????????????
-sequential_13/dense_52/BiasAdd/ReadVariableOpReadVariableOp6sequential_13_dense_52_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_13/dense_52/BiasAddBiasAdd)sequential_13/dense_52/Tensordot:output:05sequential_13/dense_52/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:????????????
sequential_13/dense_52/ReluRelu'sequential_13/dense_52/BiasAdd:output:0*
T0*-
_output_shapes
:????????????
/sequential_13/dense_53/Tensordot/ReadVariableOpReadVariableOp8sequential_13_dense_53_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype0o
%sequential_13/dense_53/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:v
%sequential_13/dense_53/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
&sequential_13/dense_53/Tensordot/ShapeShape)sequential_13/dense_52/Relu:activations:0*
T0*
_output_shapes
:p
.sequential_13/dense_53/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
)sequential_13/dense_53/Tensordot/GatherV2GatherV2/sequential_13/dense_53/Tensordot/Shape:output:0.sequential_13/dense_53/Tensordot/free:output:07sequential_13/dense_53/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:r
0sequential_13/dense_53/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+sequential_13/dense_53/Tensordot/GatherV2_1GatherV2/sequential_13/dense_53/Tensordot/Shape:output:0.sequential_13/dense_53/Tensordot/axes:output:09sequential_13/dense_53/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
&sequential_13/dense_53/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
%sequential_13/dense_53/Tensordot/ProdProd2sequential_13/dense_53/Tensordot/GatherV2:output:0/sequential_13/dense_53/Tensordot/Const:output:0*
T0*
_output_shapes
: r
(sequential_13/dense_53/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
'sequential_13/dense_53/Tensordot/Prod_1Prod4sequential_13/dense_53/Tensordot/GatherV2_1:output:01sequential_13/dense_53/Tensordot/Const_1:output:0*
T0*
_output_shapes
: n
,sequential_13/dense_53/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
'sequential_13/dense_53/Tensordot/concatConcatV2.sequential_13/dense_53/Tensordot/free:output:0.sequential_13/dense_53/Tensordot/axes:output:05sequential_13/dense_53/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
&sequential_13/dense_53/Tensordot/stackPack.sequential_13/dense_53/Tensordot/Prod:output:00sequential_13/dense_53/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
*sequential_13/dense_53/Tensordot/transpose	Transpose)sequential_13/dense_52/Relu:activations:00sequential_13/dense_53/Tensordot/concat:output:0*
T0*-
_output_shapes
:????????????
(sequential_13/dense_53/Tensordot/ReshapeReshape.sequential_13/dense_53/Tensordot/transpose:y:0/sequential_13/dense_53/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
'sequential_13/dense_53/Tensordot/MatMulMatMul1sequential_13/dense_53/Tensordot/Reshape:output:07sequential_13/dense_53/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
(sequential_13/dense_53/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?p
.sequential_13/dense_53/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
)sequential_13/dense_53/Tensordot/concat_1ConcatV22sequential_13/dense_53/Tensordot/GatherV2:output:01sequential_13/dense_53/Tensordot/Const_2:output:07sequential_13/dense_53/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
 sequential_13/dense_53/TensordotReshape1sequential_13/dense_53/Tensordot/MatMul:product:02sequential_13/dense_53/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:????????????
-sequential_13/dense_53/BiasAdd/ReadVariableOpReadVariableOp6sequential_13_dense_53_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_13/dense_53/BiasAddBiasAdd)sequential_13/dense_53/Tensordot:output:05sequential_13/dense_53/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:????????????
dropout_56/IdentityIdentity'sequential_13/dense_53/BiasAdd:output:0*
T0*-
_output_shapes
:????????????
add_1AddV2layer_normalization_26/add:z:0dropout_56/Identity:output:0*
T0*-
_output_shapes
:???????????U
layer_normalization_27/ShapeShape	add_1:z:0*
T0*
_output_shapes
:t
*layer_normalization_27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,layer_normalization_27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,layer_normalization_27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$layer_normalization_27/strided_sliceStridedSlice%layer_normalization_27/Shape:output:03layer_normalization_27/strided_slice/stack:output:05layer_normalization_27/strided_slice/stack_1:output:05layer_normalization_27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
layer_normalization_27/mul/xConst*
_output_shapes
: *
dtype0*
value	B :?
layer_normalization_27/mulMul%layer_normalization_27/mul/x:output:0-layer_normalization_27/strided_slice:output:0*
T0*
_output_shapes
: v
,layer_normalization_27/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_27/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_27/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_27/strided_slice_1StridedSlice%layer_normalization_27/Shape:output:05layer_normalization_27/strided_slice_1/stack:output:07layer_normalization_27/strided_slice_1/stack_1:output:07layer_normalization_27/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
layer_normalization_27/mul_1Mullayer_normalization_27/mul:z:0/layer_normalization_27/strided_slice_1:output:0*
T0*
_output_shapes
: v
,layer_normalization_27/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_27/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_27/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_27/strided_slice_2StridedSlice%layer_normalization_27/Shape:output:05layer_normalization_27/strided_slice_2/stack:output:07layer_normalization_27/strided_slice_2/stack_1:output:07layer_normalization_27/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
layer_normalization_27/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :?
layer_normalization_27/mul_2Mul'layer_normalization_27/mul_2/x:output:0/layer_normalization_27/strided_slice_2:output:0*
T0*
_output_shapes
: h
&layer_normalization_27/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :h
&layer_normalization_27/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
$layer_normalization_27/Reshape/shapePack/layer_normalization_27/Reshape/shape/0:output:0 layer_normalization_27/mul_1:z:0 layer_normalization_27/mul_2:z:0/layer_normalization_27/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
layer_normalization_27/ReshapeReshape	add_1:z:0-layer_normalization_27/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????e
"layer_normalization_27/ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :??
 layer_normalization_27/ones/LessLess layer_normalization_27/mul_1:z:0+layer_normalization_27/ones/Less/y:output:0*
T0*
_output_shapes
: z
"layer_normalization_27/ones/packedPack layer_normalization_27/mul_1:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_27/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
layer_normalization_27/onesFill+layer_normalization_27/ones/packed:output:0*layer_normalization_27/ones/Const:output:0*
T0*#
_output_shapes
:?????????f
#layer_normalization_27/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :??
!layer_normalization_27/zeros/LessLess layer_normalization_27/mul_1:z:0,layer_normalization_27/zeros/Less/y:output:0*
T0*
_output_shapes
: {
#layer_normalization_27/zeros/packedPack layer_normalization_27/mul_1:z:0*
N*
T0*
_output_shapes
:g
"layer_normalization_27/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
layer_normalization_27/zerosFill,layer_normalization_27/zeros/packed:output:0+layer_normalization_27/zeros/Const:output:0*
T0*#
_output_shapes
:?????????_
layer_normalization_27/ConstConst*
_output_shapes
: *
dtype0*
valueB a
layer_normalization_27/Const_1Const*
_output_shapes
: *
dtype0*
valueB ?
'layer_normalization_27/FusedBatchNormV3FusedBatchNormV3'layer_normalization_27/Reshape:output:0$layer_normalization_27/ones:output:0%layer_normalization_27/zeros:output:0%layer_normalization_27/Const:output:0'layer_normalization_27/Const_1:output:0*
T0*
U0*p
_output_shapes^
\:??????????:?????????:?????????:?????????:?????????:*
data_formatNCHW?
 layer_normalization_27/Reshape_1Reshape+layer_normalization_27/FusedBatchNormV3:y:0%layer_normalization_27/Shape:output:0*
T0*-
_output_shapes
:????????????
+layer_normalization_27/mul_3/ReadVariableOpReadVariableOp4layer_normalization_27_mul_3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
layer_normalization_27/mul_3Mul)layer_normalization_27/Reshape_1:output:03layer_normalization_27/mul_3/ReadVariableOp:value:0*
T0*-
_output_shapes
:????????????
)layer_normalization_27/add/ReadVariableOpReadVariableOp2layer_normalization_27_add_readvariableop_resource*
_output_shapes	
:?*
dtype0?
layer_normalization_27/addAddV2 layer_normalization_27/mul_3:z:01layer_normalization_27/add/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????s
IdentityIdentitylayer_normalization_27/add:z:0^NoOp*
T0*-
_output_shapes
:????????????
NoOpNoOp*^layer_normalization_26/add/ReadVariableOp,^layer_normalization_26/mul_3/ReadVariableOp*^layer_normalization_27/add/ReadVariableOp,^layer_normalization_27/mul_3/ReadVariableOp<^multi_head_attention_13/attention_output/add/ReadVariableOpF^multi_head_attention_13/attention_output/einsum/Einsum/ReadVariableOp/^multi_head_attention_13/key/add/ReadVariableOp9^multi_head_attention_13/key/einsum/Einsum/ReadVariableOp1^multi_head_attention_13/query/add/ReadVariableOp;^multi_head_attention_13/query/einsum/Einsum/ReadVariableOp1^multi_head_attention_13/value/add/ReadVariableOp;^multi_head_attention_13/value/einsum/Einsum/ReadVariableOp.^sequential_13/dense_52/BiasAdd/ReadVariableOp0^sequential_13/dense_52/Tensordot/ReadVariableOp.^sequential_13/dense_53/BiasAdd/ReadVariableOp0^sequential_13/dense_53/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:???????????: : : : : : : : : : : : : : : : 2V
)layer_normalization_26/add/ReadVariableOp)layer_normalization_26/add/ReadVariableOp2Z
+layer_normalization_26/mul_3/ReadVariableOp+layer_normalization_26/mul_3/ReadVariableOp2V
)layer_normalization_27/add/ReadVariableOp)layer_normalization_27/add/ReadVariableOp2Z
+layer_normalization_27/mul_3/ReadVariableOp+layer_normalization_27/mul_3/ReadVariableOp2z
;multi_head_attention_13/attention_output/add/ReadVariableOp;multi_head_attention_13/attention_output/add/ReadVariableOp2?
Emulti_head_attention_13/attention_output/einsum/Einsum/ReadVariableOpEmulti_head_attention_13/attention_output/einsum/Einsum/ReadVariableOp2`
.multi_head_attention_13/key/add/ReadVariableOp.multi_head_attention_13/key/add/ReadVariableOp2t
8multi_head_attention_13/key/einsum/Einsum/ReadVariableOp8multi_head_attention_13/key/einsum/Einsum/ReadVariableOp2d
0multi_head_attention_13/query/add/ReadVariableOp0multi_head_attention_13/query/add/ReadVariableOp2x
:multi_head_attention_13/query/einsum/Einsum/ReadVariableOp:multi_head_attention_13/query/einsum/Einsum/ReadVariableOp2d
0multi_head_attention_13/value/add/ReadVariableOp0multi_head_attention_13/value/add/ReadVariableOp2x
:multi_head_attention_13/value/einsum/Einsum/ReadVariableOp:multi_head_attention_13/value/einsum/Einsum/ReadVariableOp2^
-sequential_13/dense_52/BiasAdd/ReadVariableOp-sequential_13/dense_52/BiasAdd/ReadVariableOp2b
/sequential_13/dense_52/Tensordot/ReadVariableOp/sequential_13/dense_52/Tensordot/ReadVariableOp2^
-sequential_13/dense_53/BiasAdd/ReadVariableOp-sequential_13/dense_53/BiasAdd/ReadVariableOp2b
/sequential_13/dense_53/Tensordot/ReadVariableOp/sequential_13/dense_53/Tensordot/ReadVariableOp:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
(__inference_dense_54_layer_call_fn_83560

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_54_layer_call_and_return_conditional_losses_81640o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
H__inference_sequential_13_layer_call_and_return_conditional_losses_83758

inputs>
*dense_52_tensordot_readvariableop_resource:
??7
(dense_52_biasadd_readvariableop_resource:	?>
*dense_53_tensordot_readvariableop_resource:
??7
(dense_53_biasadd_readvariableop_resource:	?
identity??dense_52/BiasAdd/ReadVariableOp?!dense_52/Tensordot/ReadVariableOp?dense_53/BiasAdd/ReadVariableOp?!dense_53/Tensordot/ReadVariableOp?
!dense_52/Tensordot/ReadVariableOpReadVariableOp*dense_52_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype0a
dense_52/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_52/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       N
dense_52/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:b
 dense_52/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_52/Tensordot/GatherV2GatherV2!dense_52/Tensordot/Shape:output:0 dense_52/Tensordot/free:output:0)dense_52/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_52/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_52/Tensordot/GatherV2_1GatherV2!dense_52/Tensordot/Shape:output:0 dense_52/Tensordot/axes:output:0+dense_52/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_52/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_52/Tensordot/ProdProd$dense_52/Tensordot/GatherV2:output:0!dense_52/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_52/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_52/Tensordot/Prod_1Prod&dense_52/Tensordot/GatherV2_1:output:0#dense_52/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_52/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_52/Tensordot/concatConcatV2 dense_52/Tensordot/free:output:0 dense_52/Tensordot/axes:output:0'dense_52/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_52/Tensordot/stackPack dense_52/Tensordot/Prod:output:0"dense_52/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_52/Tensordot/transpose	Transposeinputs"dense_52/Tensordot/concat:output:0*
T0*-
_output_shapes
:????????????
dense_52/Tensordot/ReshapeReshape dense_52/Tensordot/transpose:y:0!dense_52/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_52/Tensordot/MatMulMatMul#dense_52/Tensordot/Reshape:output:0)dense_52/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????e
dense_52/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?b
 dense_52/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_52/Tensordot/concat_1ConcatV2$dense_52/Tensordot/GatherV2:output:0#dense_52/Tensordot/Const_2:output:0)dense_52/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_52/TensordotReshape#dense_52/Tensordot/MatMul:product:0$dense_52/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:????????????
dense_52/BiasAdd/ReadVariableOpReadVariableOp(dense_52_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_52/BiasAddBiasAdddense_52/Tensordot:output:0'dense_52/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????h
dense_52/ReluReludense_52/BiasAdd:output:0*
T0*-
_output_shapes
:????????????
!dense_53/Tensordot/ReadVariableOpReadVariableOp*dense_53_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype0a
dense_53/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_53/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       c
dense_53/Tensordot/ShapeShapedense_52/Relu:activations:0*
T0*
_output_shapes
:b
 dense_53/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_53/Tensordot/GatherV2GatherV2!dense_53/Tensordot/Shape:output:0 dense_53/Tensordot/free:output:0)dense_53/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_53/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_53/Tensordot/GatherV2_1GatherV2!dense_53/Tensordot/Shape:output:0 dense_53/Tensordot/axes:output:0+dense_53/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_53/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_53/Tensordot/ProdProd$dense_53/Tensordot/GatherV2:output:0!dense_53/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_53/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_53/Tensordot/Prod_1Prod&dense_53/Tensordot/GatherV2_1:output:0#dense_53/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_53/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_53/Tensordot/concatConcatV2 dense_53/Tensordot/free:output:0 dense_53/Tensordot/axes:output:0'dense_53/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_53/Tensordot/stackPack dense_53/Tensordot/Prod:output:0"dense_53/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_53/Tensordot/transpose	Transposedense_52/Relu:activations:0"dense_53/Tensordot/concat:output:0*
T0*-
_output_shapes
:????????????
dense_53/Tensordot/ReshapeReshape dense_53/Tensordot/transpose:y:0!dense_53/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_53/Tensordot/MatMulMatMul#dense_53/Tensordot/Reshape:output:0)dense_53/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????e
dense_53/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?b
 dense_53/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_53/Tensordot/concat_1ConcatV2$dense_53/Tensordot/GatherV2:output:0#dense_53/Tensordot/Const_2:output:0)dense_53/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_53/TensordotReshape#dense_53/Tensordot/MatMul:product:0$dense_53/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:????????????
dense_53/BiasAdd/ReadVariableOpReadVariableOp(dense_53_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_53/BiasAddBiasAdddense_53/Tensordot:output:0'dense_53/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????n
IdentityIdentitydense_53/BiasAdd:output:0^NoOp*
T0*-
_output_shapes
:????????????
NoOpNoOp ^dense_52/BiasAdd/ReadVariableOp"^dense_52/Tensordot/ReadVariableOp ^dense_53/BiasAdd/ReadVariableOp"^dense_53/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : : : 2B
dense_52/BiasAdd/ReadVariableOpdense_52/BiasAdd/ReadVariableOp2F
!dense_52/Tensordot/ReadVariableOp!dense_52/Tensordot/ReadVariableOp2B
dense_53/BiasAdd/ReadVariableOpdense_53/BiasAdd/ReadVariableOp2F
!dense_53/Tensordot/ReadVariableOp!dense_53/Tensordot/ReadVariableOp:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
(__inference_model_13_layer_call_fn_82539

inputs
unknown:
??
	unknown_0:
?)?!
	unknown_1:??
	unknown_2:	?!
	unknown_3:??
	unknown_4:	?!
	unknown_5:??
	unknown_6:	?!
	unknown_7:??
	unknown_8:	?
	unknown_9:	?

unknown_10:	?

unknown_11:
??

unknown_12:	?

unknown_13:
??

unknown_14:	?

unknown_15:	?

unknown_16:	?

unknown_17:	?

unknown_18:

unknown_19:

unknown_20:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_model_13_layer_call_and_return_conditional_losses_82178o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:??????????: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
W
;__inference_global_average_pooling1d_13_layer_call_fn_83512

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *_
fZRX
V__inference_global_average_pooling1d_13_layer_call_and_return_conditional_losses_81620a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
(__inference_model_13_layer_call_fn_82274
input_14
unknown:
??
	unknown_0:
?)?!
	unknown_1:??
	unknown_2:	?!
	unknown_3:??
	unknown_4:	?!
	unknown_5:??
	unknown_6:	?!
	unknown_7:??
	unknown_8:	?
	unknown_9:	?

unknown_10:	?

unknown_11:
??

unknown_12:	?

unknown_13:
??

unknown_14:	?

unknown_15:	?

unknown_16:	?

unknown_17:	?

unknown_18:

unknown_19:

unknown_20:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_14unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_model_13_layer_call_and_return_conditional_losses_82178o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:??????????: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:??????????
"
_user_specified_name
input_14
?
?
C__inference_dense_53_layer_call_and_return_conditional_losses_83837

inputs5
!tensordot_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:{
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*-
_output_shapes
:????????????
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*-
_output_shapes
:???????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????e
IdentityIdentityBiasAdd:output:0^NoOp*
T0*-
_output_shapes
:???????????z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
4__inference_transformer_block_13_layer_call_fn_83119

inputs
unknown:??
	unknown_0:	?!
	unknown_1:??
	unknown_2:	?!
	unknown_3:??
	unknown_4:	?!
	unknown_5:??
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:
??

unknown_10:	?

unknown_11:
??

unknown_12:	?

unknown_13:	?

unknown_14:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_transformer_block_13_layer_call_and_return_conditional_losses_82027u
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:???????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_57_layer_call_and_return_conditional_losses_83539

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_dense_53_layer_call_fn_83807

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_53_layer_call_and_return_conditional_losses_81227u
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
C__inference_dense_53_layer_call_and_return_conditional_losses_81227

inputs5
!tensordot_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:{
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*-
_output_shapes
:????????????
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*-
_output_shapes
:???????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????e
IdentityIdentityBiasAdd:output:0^NoOp*
T0*-
_output_shapes
:???????????z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
ܕ
?
 __inference__wrapped_model_81153
input_14`
Lmodel_13_token_and_position_embedding_13_embedding_27_embedding_lookup_80941:
??`
Lmodel_13_token_and_position_embedding_13_embedding_26_embedding_lookup_80947:
?)?y
amodel_13_transformer_block_13_multi_head_attention_13_query_einsum_einsum_readvariableop_resource:??j
Wmodel_13_transformer_block_13_multi_head_attention_13_query_add_readvariableop_resource:	?w
_model_13_transformer_block_13_multi_head_attention_13_key_einsum_einsum_readvariableop_resource:??h
Umodel_13_transformer_block_13_multi_head_attention_13_key_add_readvariableop_resource:	?y
amodel_13_transformer_block_13_multi_head_attention_13_value_einsum_einsum_readvariableop_resource:??j
Wmodel_13_transformer_block_13_multi_head_attention_13_value_add_readvariableop_resource:	??
lmodel_13_transformer_block_13_multi_head_attention_13_attention_output_einsum_einsum_readvariableop_resource:??q
bmodel_13_transformer_block_13_multi_head_attention_13_attention_output_add_readvariableop_resource:	?a
Rmodel_13_transformer_block_13_layer_normalization_26_mul_3_readvariableop_resource:	?_
Pmodel_13_transformer_block_13_layer_normalization_26_add_readvariableop_resource:	?j
Vmodel_13_transformer_block_13_sequential_13_dense_52_tensordot_readvariableop_resource:
??c
Tmodel_13_transformer_block_13_sequential_13_dense_52_biasadd_readvariableop_resource:	?j
Vmodel_13_transformer_block_13_sequential_13_dense_53_tensordot_readvariableop_resource:
??c
Tmodel_13_transformer_block_13_sequential_13_dense_53_biasadd_readvariableop_resource:	?a
Rmodel_13_transformer_block_13_layer_normalization_27_mul_3_readvariableop_resource:	?_
Pmodel_13_transformer_block_13_layer_normalization_27_add_readvariableop_resource:	?C
0model_13_dense_54_matmul_readvariableop_resource:	??
1model_13_dense_54_biasadd_readvariableop_resource:B
0model_13_dense_55_matmul_readvariableop_resource:?
1model_13_dense_55_biasadd_readvariableop_resource:
identity??(model_13/dense_54/BiasAdd/ReadVariableOp?'model_13/dense_54/MatMul/ReadVariableOp?(model_13/dense_55/BiasAdd/ReadVariableOp?'model_13/dense_55/MatMul/ReadVariableOp?Fmodel_13/token_and_position_embedding_13/embedding_26/embedding_lookup?Fmodel_13/token_and_position_embedding_13/embedding_27/embedding_lookup?Gmodel_13/transformer_block_13/layer_normalization_26/add/ReadVariableOp?Imodel_13/transformer_block_13/layer_normalization_26/mul_3/ReadVariableOp?Gmodel_13/transformer_block_13/layer_normalization_27/add/ReadVariableOp?Imodel_13/transformer_block_13/layer_normalization_27/mul_3/ReadVariableOp?Ymodel_13/transformer_block_13/multi_head_attention_13/attention_output/add/ReadVariableOp?cmodel_13/transformer_block_13/multi_head_attention_13/attention_output/einsum/Einsum/ReadVariableOp?Lmodel_13/transformer_block_13/multi_head_attention_13/key/add/ReadVariableOp?Vmodel_13/transformer_block_13/multi_head_attention_13/key/einsum/Einsum/ReadVariableOp?Nmodel_13/transformer_block_13/multi_head_attention_13/query/add/ReadVariableOp?Xmodel_13/transformer_block_13/multi_head_attention_13/query/einsum/Einsum/ReadVariableOp?Nmodel_13/transformer_block_13/multi_head_attention_13/value/add/ReadVariableOp?Xmodel_13/transformer_block_13/multi_head_attention_13/value/einsum/Einsum/ReadVariableOp?Kmodel_13/transformer_block_13/sequential_13/dense_52/BiasAdd/ReadVariableOp?Mmodel_13/transformer_block_13/sequential_13/dense_52/Tensordot/ReadVariableOp?Kmodel_13/transformer_block_13/sequential_13/dense_53/BiasAdd/ReadVariableOp?Mmodel_13/transformer_block_13/sequential_13/dense_53/Tensordot/ReadVariableOpf
.model_13/token_and_position_embedding_13/ShapeShapeinput_14*
T0*
_output_shapes
:?
<model_13/token_and_position_embedding_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
>model_13/token_and_position_embedding_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ?
>model_13/token_and_position_embedding_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
6model_13/token_and_position_embedding_13/strided_sliceStridedSlice7model_13/token_and_position_embedding_13/Shape:output:0Emodel_13/token_and_position_embedding_13/strided_slice/stack:output:0Gmodel_13/token_and_position_embedding_13/strided_slice/stack_1:output:0Gmodel_13/token_and_position_embedding_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
4model_13/token_and_position_embedding_13/range/startConst*
_output_shapes
: *
dtype0*
value	B : v
4model_13/token_and_position_embedding_13/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
.model_13/token_and_position_embedding_13/rangeRange=model_13/token_and_position_embedding_13/range/start:output:0?model_13/token_and_position_embedding_13/strided_slice:output:0=model_13/token_and_position_embedding_13/range/delta:output:0*
_output_shapes	
:??
Fmodel_13/token_and_position_embedding_13/embedding_27/embedding_lookupResourceGatherLmodel_13_token_and_position_embedding_13_embedding_27_embedding_lookup_809417model_13/token_and_position_embedding_13/range:output:0*
Tindices0*_
_classU
SQloc:@model_13/token_and_position_embedding_13/embedding_27/embedding_lookup/80941* 
_output_shapes
:
??*
dtype0?
Omodel_13/token_and_position_embedding_13/embedding_27/embedding_lookup/IdentityIdentityOmodel_13/token_and_position_embedding_13/embedding_27/embedding_lookup:output:0*
T0*_
_classU
SQloc:@model_13/token_and_position_embedding_13/embedding_27/embedding_lookup/80941* 
_output_shapes
:
???
Qmodel_13/token_and_position_embedding_13/embedding_27/embedding_lookup/Identity_1IdentityXmodel_13/token_and_position_embedding_13/embedding_27/embedding_lookup/Identity:output:0*
T0* 
_output_shapes
:
???
:model_13/token_and_position_embedding_13/embedding_26/CastCastinput_14*

DstT0*

SrcT0*(
_output_shapes
:???????????
Fmodel_13/token_and_position_embedding_13/embedding_26/embedding_lookupResourceGatherLmodel_13_token_and_position_embedding_13_embedding_26_embedding_lookup_80947>model_13/token_and_position_embedding_13/embedding_26/Cast:y:0*
Tindices0*_
_classU
SQloc:@model_13/token_and_position_embedding_13/embedding_26/embedding_lookup/80947*-
_output_shapes
:???????????*
dtype0?
Omodel_13/token_and_position_embedding_13/embedding_26/embedding_lookup/IdentityIdentityOmodel_13/token_and_position_embedding_13/embedding_26/embedding_lookup:output:0*
T0*_
_classU
SQloc:@model_13/token_and_position_embedding_13/embedding_26/embedding_lookup/80947*-
_output_shapes
:????????????
Qmodel_13/token_and_position_embedding_13/embedding_26/embedding_lookup/Identity_1IdentityXmodel_13/token_and_position_embedding_13/embedding_26/embedding_lookup/Identity:output:0*
T0*-
_output_shapes
:????????????
,model_13/token_and_position_embedding_13/addAddV2Zmodel_13/token_and_position_embedding_13/embedding_26/embedding_lookup/Identity_1:output:0Zmodel_13/token_and_position_embedding_13/embedding_27/embedding_lookup/Identity_1:output:0*
T0*-
_output_shapes
:????????????
Xmodel_13/transformer_block_13/multi_head_attention_13/query/einsum/Einsum/ReadVariableOpReadVariableOpamodel_13_transformer_block_13_multi_head_attention_13_query_einsum_einsum_readvariableop_resource*$
_output_shapes
:??*
dtype0?
Imodel_13/transformer_block_13/multi_head_attention_13/query/einsum/EinsumEinsum0model_13/token_and_position_embedding_13/add:z:0`model_13/transformer_block_13/multi_head_attention_13/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*1
_output_shapes
:???????????*
equationabc,cde->abde?
Nmodel_13/transformer_block_13/multi_head_attention_13/query/add/ReadVariableOpReadVariableOpWmodel_13_transformer_block_13_multi_head_attention_13_query_add_readvariableop_resource*
_output_shapes
:	?*
dtype0?
?model_13/transformer_block_13/multi_head_attention_13/query/addAddV2Rmodel_13/transformer_block_13/multi_head_attention_13/query/einsum/Einsum:output:0Vmodel_13/transformer_block_13/multi_head_attention_13/query/add/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
Vmodel_13/transformer_block_13/multi_head_attention_13/key/einsum/Einsum/ReadVariableOpReadVariableOp_model_13_transformer_block_13_multi_head_attention_13_key_einsum_einsum_readvariableop_resource*$
_output_shapes
:??*
dtype0?
Gmodel_13/transformer_block_13/multi_head_attention_13/key/einsum/EinsumEinsum0model_13/token_and_position_embedding_13/add:z:0^model_13/transformer_block_13/multi_head_attention_13/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*1
_output_shapes
:???????????*
equationabc,cde->abde?
Lmodel_13/transformer_block_13/multi_head_attention_13/key/add/ReadVariableOpReadVariableOpUmodel_13_transformer_block_13_multi_head_attention_13_key_add_readvariableop_resource*
_output_shapes
:	?*
dtype0?
=model_13/transformer_block_13/multi_head_attention_13/key/addAddV2Pmodel_13/transformer_block_13/multi_head_attention_13/key/einsum/Einsum:output:0Tmodel_13/transformer_block_13/multi_head_attention_13/key/add/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
Xmodel_13/transformer_block_13/multi_head_attention_13/value/einsum/Einsum/ReadVariableOpReadVariableOpamodel_13_transformer_block_13_multi_head_attention_13_value_einsum_einsum_readvariableop_resource*$
_output_shapes
:??*
dtype0?
Imodel_13/transformer_block_13/multi_head_attention_13/value/einsum/EinsumEinsum0model_13/token_and_position_embedding_13/add:z:0`model_13/transformer_block_13/multi_head_attention_13/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*1
_output_shapes
:???????????*
equationabc,cde->abde?
Nmodel_13/transformer_block_13/multi_head_attention_13/value/add/ReadVariableOpReadVariableOpWmodel_13_transformer_block_13_multi_head_attention_13_value_add_readvariableop_resource*
_output_shapes
:	?*
dtype0?
?model_13/transformer_block_13/multi_head_attention_13/value/addAddV2Rmodel_13/transformer_block_13/multi_head_attention_13/value/einsum/Einsum:output:0Vmodel_13/transformer_block_13/multi_head_attention_13/value/add/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
;model_13/transformer_block_13/multi_head_attention_13/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *?А=?
9model_13/transformer_block_13/multi_head_attention_13/MulMulCmodel_13/transformer_block_13/multi_head_attention_13/query/add:z:0Dmodel_13/transformer_block_13/multi_head_attention_13/Mul/y:output:0*
T0*1
_output_shapes
:????????????
Cmodel_13/transformer_block_13/multi_head_attention_13/einsum/EinsumEinsumAmodel_13/transformer_block_13/multi_head_attention_13/key/add:z:0=model_13/transformer_block_13/multi_head_attention_13/Mul:z:0*
N*
T0*1
_output_shapes
:???????????*
equationaecd,abcd->acbe?
Emodel_13/transformer_block_13/multi_head_attention_13/softmax/SoftmaxSoftmaxLmodel_13/transformer_block_13/multi_head_attention_13/einsum/Einsum:output:0*
T0*1
_output_shapes
:????????????
Fmodel_13/transformer_block_13/multi_head_attention_13/dropout/IdentityIdentityOmodel_13/transformer_block_13/multi_head_attention_13/softmax/Softmax:softmax:0*
T0*1
_output_shapes
:????????????
Emodel_13/transformer_block_13/multi_head_attention_13/einsum_1/EinsumEinsumOmodel_13/transformer_block_13/multi_head_attention_13/dropout/Identity:output:0Cmodel_13/transformer_block_13/multi_head_attention_13/value/add:z:0*
N*
T0*1
_output_shapes
:???????????*
equationacbe,aecd->abcd?
cmodel_13/transformer_block_13/multi_head_attention_13/attention_output/einsum/Einsum/ReadVariableOpReadVariableOplmodel_13_transformer_block_13_multi_head_attention_13_attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:??*
dtype0?
Tmodel_13/transformer_block_13/multi_head_attention_13/attention_output/einsum/EinsumEinsumNmodel_13/transformer_block_13/multi_head_attention_13/einsum_1/Einsum:output:0kmodel_13/transformer_block_13/multi_head_attention_13/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*-
_output_shapes
:???????????*
equationabcd,cde->abe?
Ymodel_13/transformer_block_13/multi_head_attention_13/attention_output/add/ReadVariableOpReadVariableOpbmodel_13_transformer_block_13_multi_head_attention_13_attention_output_add_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Jmodel_13/transformer_block_13/multi_head_attention_13/attention_output/addAddV2]model_13/transformer_block_13/multi_head_attention_13/attention_output/einsum/Einsum:output:0amodel_13/transformer_block_13/multi_head_attention_13/attention_output/add/ReadVariableOp:value:0*
T0*-
_output_shapes
:????????????
1model_13/transformer_block_13/dropout_55/IdentityIdentityNmodel_13/transformer_block_13/multi_head_attention_13/attention_output/add:z:0*
T0*-
_output_shapes
:????????????
!model_13/transformer_block_13/addAddV20model_13/token_and_position_embedding_13/add:z:0:model_13/transformer_block_13/dropout_55/Identity:output:0*
T0*-
_output_shapes
:????????????
:model_13/transformer_block_13/layer_normalization_26/ShapeShape%model_13/transformer_block_13/add:z:0*
T0*
_output_shapes
:?
Hmodel_13/transformer_block_13/layer_normalization_26/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Jmodel_13/transformer_block_13/layer_normalization_26/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Jmodel_13/transformer_block_13/layer_normalization_26/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Bmodel_13/transformer_block_13/layer_normalization_26/strided_sliceStridedSliceCmodel_13/transformer_block_13/layer_normalization_26/Shape:output:0Qmodel_13/transformer_block_13/layer_normalization_26/strided_slice/stack:output:0Smodel_13/transformer_block_13/layer_normalization_26/strided_slice/stack_1:output:0Smodel_13/transformer_block_13/layer_normalization_26/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
:model_13/transformer_block_13/layer_normalization_26/mul/xConst*
_output_shapes
: *
dtype0*
value	B :?
8model_13/transformer_block_13/layer_normalization_26/mulMulCmodel_13/transformer_block_13/layer_normalization_26/mul/x:output:0Kmodel_13/transformer_block_13/layer_normalization_26/strided_slice:output:0*
T0*
_output_shapes
: ?
Jmodel_13/transformer_block_13/layer_normalization_26/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Lmodel_13/transformer_block_13/layer_normalization_26/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Lmodel_13/transformer_block_13/layer_normalization_26/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Dmodel_13/transformer_block_13/layer_normalization_26/strided_slice_1StridedSliceCmodel_13/transformer_block_13/layer_normalization_26/Shape:output:0Smodel_13/transformer_block_13/layer_normalization_26/strided_slice_1/stack:output:0Umodel_13/transformer_block_13/layer_normalization_26/strided_slice_1/stack_1:output:0Umodel_13/transformer_block_13/layer_normalization_26/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
:model_13/transformer_block_13/layer_normalization_26/mul_1Mul<model_13/transformer_block_13/layer_normalization_26/mul:z:0Mmodel_13/transformer_block_13/layer_normalization_26/strided_slice_1:output:0*
T0*
_output_shapes
: ?
Jmodel_13/transformer_block_13/layer_normalization_26/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Lmodel_13/transformer_block_13/layer_normalization_26/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Lmodel_13/transformer_block_13/layer_normalization_26/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Dmodel_13/transformer_block_13/layer_normalization_26/strided_slice_2StridedSliceCmodel_13/transformer_block_13/layer_normalization_26/Shape:output:0Smodel_13/transformer_block_13/layer_normalization_26/strided_slice_2/stack:output:0Umodel_13/transformer_block_13/layer_normalization_26/strided_slice_2/stack_1:output:0Umodel_13/transformer_block_13/layer_normalization_26/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask~
<model_13/transformer_block_13/layer_normalization_26/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :?
:model_13/transformer_block_13/layer_normalization_26/mul_2MulEmodel_13/transformer_block_13/layer_normalization_26/mul_2/x:output:0Mmodel_13/transformer_block_13/layer_normalization_26/strided_slice_2:output:0*
T0*
_output_shapes
: ?
Dmodel_13/transformer_block_13/layer_normalization_26/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :?
Dmodel_13/transformer_block_13/layer_normalization_26/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
Bmodel_13/transformer_block_13/layer_normalization_26/Reshape/shapePackMmodel_13/transformer_block_13/layer_normalization_26/Reshape/shape/0:output:0>model_13/transformer_block_13/layer_normalization_26/mul_1:z:0>model_13/transformer_block_13/layer_normalization_26/mul_2:z:0Mmodel_13/transformer_block_13/layer_normalization_26/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
<model_13/transformer_block_13/layer_normalization_26/ReshapeReshape%model_13/transformer_block_13/add:z:0Kmodel_13/transformer_block_13/layer_normalization_26/Reshape/shape:output:0*
T0*0
_output_shapes
:???????????
@model_13/transformer_block_13/layer_normalization_26/ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :??
>model_13/transformer_block_13/layer_normalization_26/ones/LessLess>model_13/transformer_block_13/layer_normalization_26/mul_1:z:0Imodel_13/transformer_block_13/layer_normalization_26/ones/Less/y:output:0*
T0*
_output_shapes
: ?
@model_13/transformer_block_13/layer_normalization_26/ones/packedPack>model_13/transformer_block_13/layer_normalization_26/mul_1:z:0*
N*
T0*
_output_shapes
:?
?model_13/transformer_block_13/layer_normalization_26/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
9model_13/transformer_block_13/layer_normalization_26/onesFillImodel_13/transformer_block_13/layer_normalization_26/ones/packed:output:0Hmodel_13/transformer_block_13/layer_normalization_26/ones/Const:output:0*
T0*#
_output_shapes
:??????????
Amodel_13/transformer_block_13/layer_normalization_26/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :??
?model_13/transformer_block_13/layer_normalization_26/zeros/LessLess>model_13/transformer_block_13/layer_normalization_26/mul_1:z:0Jmodel_13/transformer_block_13/layer_normalization_26/zeros/Less/y:output:0*
T0*
_output_shapes
: ?
Amodel_13/transformer_block_13/layer_normalization_26/zeros/packedPack>model_13/transformer_block_13/layer_normalization_26/mul_1:z:0*
N*
T0*
_output_shapes
:?
@model_13/transformer_block_13/layer_normalization_26/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
:model_13/transformer_block_13/layer_normalization_26/zerosFillJmodel_13/transformer_block_13/layer_normalization_26/zeros/packed:output:0Imodel_13/transformer_block_13/layer_normalization_26/zeros/Const:output:0*
T0*#
_output_shapes
:?????????}
:model_13/transformer_block_13/layer_normalization_26/ConstConst*
_output_shapes
: *
dtype0*
valueB 
<model_13/transformer_block_13/layer_normalization_26/Const_1Const*
_output_shapes
: *
dtype0*
valueB ?
Emodel_13/transformer_block_13/layer_normalization_26/FusedBatchNormV3FusedBatchNormV3Emodel_13/transformer_block_13/layer_normalization_26/Reshape:output:0Bmodel_13/transformer_block_13/layer_normalization_26/ones:output:0Cmodel_13/transformer_block_13/layer_normalization_26/zeros:output:0Cmodel_13/transformer_block_13/layer_normalization_26/Const:output:0Emodel_13/transformer_block_13/layer_normalization_26/Const_1:output:0*
T0*
U0*p
_output_shapes^
\:??????????:?????????:?????????:?????????:?????????:*
data_formatNCHW?
>model_13/transformer_block_13/layer_normalization_26/Reshape_1ReshapeImodel_13/transformer_block_13/layer_normalization_26/FusedBatchNormV3:y:0Cmodel_13/transformer_block_13/layer_normalization_26/Shape:output:0*
T0*-
_output_shapes
:????????????
Imodel_13/transformer_block_13/layer_normalization_26/mul_3/ReadVariableOpReadVariableOpRmodel_13_transformer_block_13_layer_normalization_26_mul_3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
:model_13/transformer_block_13/layer_normalization_26/mul_3MulGmodel_13/transformer_block_13/layer_normalization_26/Reshape_1:output:0Qmodel_13/transformer_block_13/layer_normalization_26/mul_3/ReadVariableOp:value:0*
T0*-
_output_shapes
:????????????
Gmodel_13/transformer_block_13/layer_normalization_26/add/ReadVariableOpReadVariableOpPmodel_13_transformer_block_13_layer_normalization_26_add_readvariableop_resource*
_output_shapes	
:?*
dtype0?
8model_13/transformer_block_13/layer_normalization_26/addAddV2>model_13/transformer_block_13/layer_normalization_26/mul_3:z:0Omodel_13/transformer_block_13/layer_normalization_26/add/ReadVariableOp:value:0*
T0*-
_output_shapes
:????????????
Mmodel_13/transformer_block_13/sequential_13/dense_52/Tensordot/ReadVariableOpReadVariableOpVmodel_13_transformer_block_13_sequential_13_dense_52_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
Cmodel_13/transformer_block_13/sequential_13/dense_52/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
Cmodel_13/transformer_block_13/sequential_13/dense_52/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
Dmodel_13/transformer_block_13/sequential_13/dense_52/Tensordot/ShapeShape<model_13/transformer_block_13/layer_normalization_26/add:z:0*
T0*
_output_shapes
:?
Lmodel_13/transformer_block_13/sequential_13/dense_52/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Gmodel_13/transformer_block_13/sequential_13/dense_52/Tensordot/GatherV2GatherV2Mmodel_13/transformer_block_13/sequential_13/dense_52/Tensordot/Shape:output:0Lmodel_13/transformer_block_13/sequential_13/dense_52/Tensordot/free:output:0Umodel_13/transformer_block_13/sequential_13/dense_52/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Nmodel_13/transformer_block_13/sequential_13/dense_52/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Imodel_13/transformer_block_13/sequential_13/dense_52/Tensordot/GatherV2_1GatherV2Mmodel_13/transformer_block_13/sequential_13/dense_52/Tensordot/Shape:output:0Lmodel_13/transformer_block_13/sequential_13/dense_52/Tensordot/axes:output:0Wmodel_13/transformer_block_13/sequential_13/dense_52/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Dmodel_13/transformer_block_13/sequential_13/dense_52/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Cmodel_13/transformer_block_13/sequential_13/dense_52/Tensordot/ProdProdPmodel_13/transformer_block_13/sequential_13/dense_52/Tensordot/GatherV2:output:0Mmodel_13/transformer_block_13/sequential_13/dense_52/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
Fmodel_13/transformer_block_13/sequential_13/dense_52/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Emodel_13/transformer_block_13/sequential_13/dense_52/Tensordot/Prod_1ProdRmodel_13/transformer_block_13/sequential_13/dense_52/Tensordot/GatherV2_1:output:0Omodel_13/transformer_block_13/sequential_13/dense_52/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ?
Jmodel_13/transformer_block_13/sequential_13/dense_52/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Emodel_13/transformer_block_13/sequential_13/dense_52/Tensordot/concatConcatV2Lmodel_13/transformer_block_13/sequential_13/dense_52/Tensordot/free:output:0Lmodel_13/transformer_block_13/sequential_13/dense_52/Tensordot/axes:output:0Smodel_13/transformer_block_13/sequential_13/dense_52/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Dmodel_13/transformer_block_13/sequential_13/dense_52/Tensordot/stackPackLmodel_13/transformer_block_13/sequential_13/dense_52/Tensordot/Prod:output:0Nmodel_13/transformer_block_13/sequential_13/dense_52/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
Hmodel_13/transformer_block_13/sequential_13/dense_52/Tensordot/transpose	Transpose<model_13/transformer_block_13/layer_normalization_26/add:z:0Nmodel_13/transformer_block_13/sequential_13/dense_52/Tensordot/concat:output:0*
T0*-
_output_shapes
:????????????
Fmodel_13/transformer_block_13/sequential_13/dense_52/Tensordot/ReshapeReshapeLmodel_13/transformer_block_13/sequential_13/dense_52/Tensordot/transpose:y:0Mmodel_13/transformer_block_13/sequential_13/dense_52/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Emodel_13/transformer_block_13/sequential_13/dense_52/Tensordot/MatMulMatMulOmodel_13/transformer_block_13/sequential_13/dense_52/Tensordot/Reshape:output:0Umodel_13/transformer_block_13/sequential_13/dense_52/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
Fmodel_13/transformer_block_13/sequential_13/dense_52/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:??
Lmodel_13/transformer_block_13/sequential_13/dense_52/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Gmodel_13/transformer_block_13/sequential_13/dense_52/Tensordot/concat_1ConcatV2Pmodel_13/transformer_block_13/sequential_13/dense_52/Tensordot/GatherV2:output:0Omodel_13/transformer_block_13/sequential_13/dense_52/Tensordot/Const_2:output:0Umodel_13/transformer_block_13/sequential_13/dense_52/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
>model_13/transformer_block_13/sequential_13/dense_52/TensordotReshapeOmodel_13/transformer_block_13/sequential_13/dense_52/Tensordot/MatMul:product:0Pmodel_13/transformer_block_13/sequential_13/dense_52/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:????????????
Kmodel_13/transformer_block_13/sequential_13/dense_52/BiasAdd/ReadVariableOpReadVariableOpTmodel_13_transformer_block_13_sequential_13_dense_52_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
<model_13/transformer_block_13/sequential_13/dense_52/BiasAddBiasAddGmodel_13/transformer_block_13/sequential_13/dense_52/Tensordot:output:0Smodel_13/transformer_block_13/sequential_13/dense_52/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:????????????
9model_13/transformer_block_13/sequential_13/dense_52/ReluReluEmodel_13/transformer_block_13/sequential_13/dense_52/BiasAdd:output:0*
T0*-
_output_shapes
:????????????
Mmodel_13/transformer_block_13/sequential_13/dense_53/Tensordot/ReadVariableOpReadVariableOpVmodel_13_transformer_block_13_sequential_13_dense_53_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
Cmodel_13/transformer_block_13/sequential_13/dense_53/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
Cmodel_13/transformer_block_13/sequential_13/dense_53/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
Dmodel_13/transformer_block_13/sequential_13/dense_53/Tensordot/ShapeShapeGmodel_13/transformer_block_13/sequential_13/dense_52/Relu:activations:0*
T0*
_output_shapes
:?
Lmodel_13/transformer_block_13/sequential_13/dense_53/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Gmodel_13/transformer_block_13/sequential_13/dense_53/Tensordot/GatherV2GatherV2Mmodel_13/transformer_block_13/sequential_13/dense_53/Tensordot/Shape:output:0Lmodel_13/transformer_block_13/sequential_13/dense_53/Tensordot/free:output:0Umodel_13/transformer_block_13/sequential_13/dense_53/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Nmodel_13/transformer_block_13/sequential_13/dense_53/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Imodel_13/transformer_block_13/sequential_13/dense_53/Tensordot/GatherV2_1GatherV2Mmodel_13/transformer_block_13/sequential_13/dense_53/Tensordot/Shape:output:0Lmodel_13/transformer_block_13/sequential_13/dense_53/Tensordot/axes:output:0Wmodel_13/transformer_block_13/sequential_13/dense_53/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Dmodel_13/transformer_block_13/sequential_13/dense_53/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Cmodel_13/transformer_block_13/sequential_13/dense_53/Tensordot/ProdProdPmodel_13/transformer_block_13/sequential_13/dense_53/Tensordot/GatherV2:output:0Mmodel_13/transformer_block_13/sequential_13/dense_53/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
Fmodel_13/transformer_block_13/sequential_13/dense_53/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Emodel_13/transformer_block_13/sequential_13/dense_53/Tensordot/Prod_1ProdRmodel_13/transformer_block_13/sequential_13/dense_53/Tensordot/GatherV2_1:output:0Omodel_13/transformer_block_13/sequential_13/dense_53/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ?
Jmodel_13/transformer_block_13/sequential_13/dense_53/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Emodel_13/transformer_block_13/sequential_13/dense_53/Tensordot/concatConcatV2Lmodel_13/transformer_block_13/sequential_13/dense_53/Tensordot/free:output:0Lmodel_13/transformer_block_13/sequential_13/dense_53/Tensordot/axes:output:0Smodel_13/transformer_block_13/sequential_13/dense_53/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Dmodel_13/transformer_block_13/sequential_13/dense_53/Tensordot/stackPackLmodel_13/transformer_block_13/sequential_13/dense_53/Tensordot/Prod:output:0Nmodel_13/transformer_block_13/sequential_13/dense_53/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
Hmodel_13/transformer_block_13/sequential_13/dense_53/Tensordot/transpose	TransposeGmodel_13/transformer_block_13/sequential_13/dense_52/Relu:activations:0Nmodel_13/transformer_block_13/sequential_13/dense_53/Tensordot/concat:output:0*
T0*-
_output_shapes
:????????????
Fmodel_13/transformer_block_13/sequential_13/dense_53/Tensordot/ReshapeReshapeLmodel_13/transformer_block_13/sequential_13/dense_53/Tensordot/transpose:y:0Mmodel_13/transformer_block_13/sequential_13/dense_53/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Emodel_13/transformer_block_13/sequential_13/dense_53/Tensordot/MatMulMatMulOmodel_13/transformer_block_13/sequential_13/dense_53/Tensordot/Reshape:output:0Umodel_13/transformer_block_13/sequential_13/dense_53/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
Fmodel_13/transformer_block_13/sequential_13/dense_53/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:??
Lmodel_13/transformer_block_13/sequential_13/dense_53/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Gmodel_13/transformer_block_13/sequential_13/dense_53/Tensordot/concat_1ConcatV2Pmodel_13/transformer_block_13/sequential_13/dense_53/Tensordot/GatherV2:output:0Omodel_13/transformer_block_13/sequential_13/dense_53/Tensordot/Const_2:output:0Umodel_13/transformer_block_13/sequential_13/dense_53/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
>model_13/transformer_block_13/sequential_13/dense_53/TensordotReshapeOmodel_13/transformer_block_13/sequential_13/dense_53/Tensordot/MatMul:product:0Pmodel_13/transformer_block_13/sequential_13/dense_53/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:????????????
Kmodel_13/transformer_block_13/sequential_13/dense_53/BiasAdd/ReadVariableOpReadVariableOpTmodel_13_transformer_block_13_sequential_13_dense_53_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
<model_13/transformer_block_13/sequential_13/dense_53/BiasAddBiasAddGmodel_13/transformer_block_13/sequential_13/dense_53/Tensordot:output:0Smodel_13/transformer_block_13/sequential_13/dense_53/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:????????????
1model_13/transformer_block_13/dropout_56/IdentityIdentityEmodel_13/transformer_block_13/sequential_13/dense_53/BiasAdd:output:0*
T0*-
_output_shapes
:????????????
#model_13/transformer_block_13/add_1AddV2<model_13/transformer_block_13/layer_normalization_26/add:z:0:model_13/transformer_block_13/dropout_56/Identity:output:0*
T0*-
_output_shapes
:????????????
:model_13/transformer_block_13/layer_normalization_27/ShapeShape'model_13/transformer_block_13/add_1:z:0*
T0*
_output_shapes
:?
Hmodel_13/transformer_block_13/layer_normalization_27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Jmodel_13/transformer_block_13/layer_normalization_27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Jmodel_13/transformer_block_13/layer_normalization_27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Bmodel_13/transformer_block_13/layer_normalization_27/strided_sliceStridedSliceCmodel_13/transformer_block_13/layer_normalization_27/Shape:output:0Qmodel_13/transformer_block_13/layer_normalization_27/strided_slice/stack:output:0Smodel_13/transformer_block_13/layer_normalization_27/strided_slice/stack_1:output:0Smodel_13/transformer_block_13/layer_normalization_27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
:model_13/transformer_block_13/layer_normalization_27/mul/xConst*
_output_shapes
: *
dtype0*
value	B :?
8model_13/transformer_block_13/layer_normalization_27/mulMulCmodel_13/transformer_block_13/layer_normalization_27/mul/x:output:0Kmodel_13/transformer_block_13/layer_normalization_27/strided_slice:output:0*
T0*
_output_shapes
: ?
Jmodel_13/transformer_block_13/layer_normalization_27/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Lmodel_13/transformer_block_13/layer_normalization_27/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Lmodel_13/transformer_block_13/layer_normalization_27/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Dmodel_13/transformer_block_13/layer_normalization_27/strided_slice_1StridedSliceCmodel_13/transformer_block_13/layer_normalization_27/Shape:output:0Smodel_13/transformer_block_13/layer_normalization_27/strided_slice_1/stack:output:0Umodel_13/transformer_block_13/layer_normalization_27/strided_slice_1/stack_1:output:0Umodel_13/transformer_block_13/layer_normalization_27/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
:model_13/transformer_block_13/layer_normalization_27/mul_1Mul<model_13/transformer_block_13/layer_normalization_27/mul:z:0Mmodel_13/transformer_block_13/layer_normalization_27/strided_slice_1:output:0*
T0*
_output_shapes
: ?
Jmodel_13/transformer_block_13/layer_normalization_27/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Lmodel_13/transformer_block_13/layer_normalization_27/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Lmodel_13/transformer_block_13/layer_normalization_27/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Dmodel_13/transformer_block_13/layer_normalization_27/strided_slice_2StridedSliceCmodel_13/transformer_block_13/layer_normalization_27/Shape:output:0Smodel_13/transformer_block_13/layer_normalization_27/strided_slice_2/stack:output:0Umodel_13/transformer_block_13/layer_normalization_27/strided_slice_2/stack_1:output:0Umodel_13/transformer_block_13/layer_normalization_27/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask~
<model_13/transformer_block_13/layer_normalization_27/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :?
:model_13/transformer_block_13/layer_normalization_27/mul_2MulEmodel_13/transformer_block_13/layer_normalization_27/mul_2/x:output:0Mmodel_13/transformer_block_13/layer_normalization_27/strided_slice_2:output:0*
T0*
_output_shapes
: ?
Dmodel_13/transformer_block_13/layer_normalization_27/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :?
Dmodel_13/transformer_block_13/layer_normalization_27/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
Bmodel_13/transformer_block_13/layer_normalization_27/Reshape/shapePackMmodel_13/transformer_block_13/layer_normalization_27/Reshape/shape/0:output:0>model_13/transformer_block_13/layer_normalization_27/mul_1:z:0>model_13/transformer_block_13/layer_normalization_27/mul_2:z:0Mmodel_13/transformer_block_13/layer_normalization_27/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
<model_13/transformer_block_13/layer_normalization_27/ReshapeReshape'model_13/transformer_block_13/add_1:z:0Kmodel_13/transformer_block_13/layer_normalization_27/Reshape/shape:output:0*
T0*0
_output_shapes
:???????????
@model_13/transformer_block_13/layer_normalization_27/ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :??
>model_13/transformer_block_13/layer_normalization_27/ones/LessLess>model_13/transformer_block_13/layer_normalization_27/mul_1:z:0Imodel_13/transformer_block_13/layer_normalization_27/ones/Less/y:output:0*
T0*
_output_shapes
: ?
@model_13/transformer_block_13/layer_normalization_27/ones/packedPack>model_13/transformer_block_13/layer_normalization_27/mul_1:z:0*
N*
T0*
_output_shapes
:?
?model_13/transformer_block_13/layer_normalization_27/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
9model_13/transformer_block_13/layer_normalization_27/onesFillImodel_13/transformer_block_13/layer_normalization_27/ones/packed:output:0Hmodel_13/transformer_block_13/layer_normalization_27/ones/Const:output:0*
T0*#
_output_shapes
:??????????
Amodel_13/transformer_block_13/layer_normalization_27/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :??
?model_13/transformer_block_13/layer_normalization_27/zeros/LessLess>model_13/transformer_block_13/layer_normalization_27/mul_1:z:0Jmodel_13/transformer_block_13/layer_normalization_27/zeros/Less/y:output:0*
T0*
_output_shapes
: ?
Amodel_13/transformer_block_13/layer_normalization_27/zeros/packedPack>model_13/transformer_block_13/layer_normalization_27/mul_1:z:0*
N*
T0*
_output_shapes
:?
@model_13/transformer_block_13/layer_normalization_27/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
:model_13/transformer_block_13/layer_normalization_27/zerosFillJmodel_13/transformer_block_13/layer_normalization_27/zeros/packed:output:0Imodel_13/transformer_block_13/layer_normalization_27/zeros/Const:output:0*
T0*#
_output_shapes
:?????????}
:model_13/transformer_block_13/layer_normalization_27/ConstConst*
_output_shapes
: *
dtype0*
valueB 
<model_13/transformer_block_13/layer_normalization_27/Const_1Const*
_output_shapes
: *
dtype0*
valueB ?
Emodel_13/transformer_block_13/layer_normalization_27/FusedBatchNormV3FusedBatchNormV3Emodel_13/transformer_block_13/layer_normalization_27/Reshape:output:0Bmodel_13/transformer_block_13/layer_normalization_27/ones:output:0Cmodel_13/transformer_block_13/layer_normalization_27/zeros:output:0Cmodel_13/transformer_block_13/layer_normalization_27/Const:output:0Emodel_13/transformer_block_13/layer_normalization_27/Const_1:output:0*
T0*
U0*p
_output_shapes^
\:??????????:?????????:?????????:?????????:?????????:*
data_formatNCHW?
>model_13/transformer_block_13/layer_normalization_27/Reshape_1ReshapeImodel_13/transformer_block_13/layer_normalization_27/FusedBatchNormV3:y:0Cmodel_13/transformer_block_13/layer_normalization_27/Shape:output:0*
T0*-
_output_shapes
:????????????
Imodel_13/transformer_block_13/layer_normalization_27/mul_3/ReadVariableOpReadVariableOpRmodel_13_transformer_block_13_layer_normalization_27_mul_3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
:model_13/transformer_block_13/layer_normalization_27/mul_3MulGmodel_13/transformer_block_13/layer_normalization_27/Reshape_1:output:0Qmodel_13/transformer_block_13/layer_normalization_27/mul_3/ReadVariableOp:value:0*
T0*-
_output_shapes
:????????????
Gmodel_13/transformer_block_13/layer_normalization_27/add/ReadVariableOpReadVariableOpPmodel_13_transformer_block_13_layer_normalization_27_add_readvariableop_resource*
_output_shapes	
:?*
dtype0?
8model_13/transformer_block_13/layer_normalization_27/addAddV2>model_13/transformer_block_13/layer_normalization_27/mul_3:z:0Omodel_13/transformer_block_13/layer_normalization_27/add/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????}
;model_13/global_average_pooling1d_13/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
)model_13/global_average_pooling1d_13/MeanMean<model_13/transformer_block_13/layer_normalization_27/add:z:0Dmodel_13/global_average_pooling1d_13/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:???????????
model_13/dropout_57/IdentityIdentity2model_13/global_average_pooling1d_13/Mean:output:0*
T0*(
_output_shapes
:???????????
'model_13/dense_54/MatMul/ReadVariableOpReadVariableOp0model_13_dense_54_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
model_13/dense_54/MatMulMatMul%model_13/dropout_57/Identity:output:0/model_13/dense_54/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
(model_13/dense_54/BiasAdd/ReadVariableOpReadVariableOp1model_13_dense_54_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_13/dense_54/BiasAddBiasAdd"model_13/dense_54/MatMul:product:00model_13/dense_54/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????t
model_13/dense_54/ReluRelu"model_13/dense_54/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
model_13/dropout_58/IdentityIdentity$model_13/dense_54/Relu:activations:0*
T0*'
_output_shapes
:??????????
'model_13/dense_55/MatMul/ReadVariableOpReadVariableOp0model_13_dense_55_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
model_13/dense_55/MatMulMatMul%model_13/dropout_58/Identity:output:0/model_13/dense_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
(model_13/dense_55/BiasAdd/ReadVariableOpReadVariableOp1model_13_dense_55_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_13/dense_55/BiasAddBiasAdd"model_13/dense_55/MatMul:product:00model_13/dense_55/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????z
model_13/dense_55/SoftmaxSoftmax"model_13/dense_55/BiasAdd:output:0*
T0*'
_output_shapes
:?????????r
IdentityIdentity#model_13/dense_55/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp)^model_13/dense_54/BiasAdd/ReadVariableOp(^model_13/dense_54/MatMul/ReadVariableOp)^model_13/dense_55/BiasAdd/ReadVariableOp(^model_13/dense_55/MatMul/ReadVariableOpG^model_13/token_and_position_embedding_13/embedding_26/embedding_lookupG^model_13/token_and_position_embedding_13/embedding_27/embedding_lookupH^model_13/transformer_block_13/layer_normalization_26/add/ReadVariableOpJ^model_13/transformer_block_13/layer_normalization_26/mul_3/ReadVariableOpH^model_13/transformer_block_13/layer_normalization_27/add/ReadVariableOpJ^model_13/transformer_block_13/layer_normalization_27/mul_3/ReadVariableOpZ^model_13/transformer_block_13/multi_head_attention_13/attention_output/add/ReadVariableOpd^model_13/transformer_block_13/multi_head_attention_13/attention_output/einsum/Einsum/ReadVariableOpM^model_13/transformer_block_13/multi_head_attention_13/key/add/ReadVariableOpW^model_13/transformer_block_13/multi_head_attention_13/key/einsum/Einsum/ReadVariableOpO^model_13/transformer_block_13/multi_head_attention_13/query/add/ReadVariableOpY^model_13/transformer_block_13/multi_head_attention_13/query/einsum/Einsum/ReadVariableOpO^model_13/transformer_block_13/multi_head_attention_13/value/add/ReadVariableOpY^model_13/transformer_block_13/multi_head_attention_13/value/einsum/Einsum/ReadVariableOpL^model_13/transformer_block_13/sequential_13/dense_52/BiasAdd/ReadVariableOpN^model_13/transformer_block_13/sequential_13/dense_52/Tensordot/ReadVariableOpL^model_13/transformer_block_13/sequential_13/dense_53/BiasAdd/ReadVariableOpN^model_13/transformer_block_13/sequential_13/dense_53/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:??????????: : : : : : : : : : : : : : : : : : : : : : 2T
(model_13/dense_54/BiasAdd/ReadVariableOp(model_13/dense_54/BiasAdd/ReadVariableOp2R
'model_13/dense_54/MatMul/ReadVariableOp'model_13/dense_54/MatMul/ReadVariableOp2T
(model_13/dense_55/BiasAdd/ReadVariableOp(model_13/dense_55/BiasAdd/ReadVariableOp2R
'model_13/dense_55/MatMul/ReadVariableOp'model_13/dense_55/MatMul/ReadVariableOp2?
Fmodel_13/token_and_position_embedding_13/embedding_26/embedding_lookupFmodel_13/token_and_position_embedding_13/embedding_26/embedding_lookup2?
Fmodel_13/token_and_position_embedding_13/embedding_27/embedding_lookupFmodel_13/token_and_position_embedding_13/embedding_27/embedding_lookup2?
Gmodel_13/transformer_block_13/layer_normalization_26/add/ReadVariableOpGmodel_13/transformer_block_13/layer_normalization_26/add/ReadVariableOp2?
Imodel_13/transformer_block_13/layer_normalization_26/mul_3/ReadVariableOpImodel_13/transformer_block_13/layer_normalization_26/mul_3/ReadVariableOp2?
Gmodel_13/transformer_block_13/layer_normalization_27/add/ReadVariableOpGmodel_13/transformer_block_13/layer_normalization_27/add/ReadVariableOp2?
Imodel_13/transformer_block_13/layer_normalization_27/mul_3/ReadVariableOpImodel_13/transformer_block_13/layer_normalization_27/mul_3/ReadVariableOp2?
Ymodel_13/transformer_block_13/multi_head_attention_13/attention_output/add/ReadVariableOpYmodel_13/transformer_block_13/multi_head_attention_13/attention_output/add/ReadVariableOp2?
cmodel_13/transformer_block_13/multi_head_attention_13/attention_output/einsum/Einsum/ReadVariableOpcmodel_13/transformer_block_13/multi_head_attention_13/attention_output/einsum/Einsum/ReadVariableOp2?
Lmodel_13/transformer_block_13/multi_head_attention_13/key/add/ReadVariableOpLmodel_13/transformer_block_13/multi_head_attention_13/key/add/ReadVariableOp2?
Vmodel_13/transformer_block_13/multi_head_attention_13/key/einsum/Einsum/ReadVariableOpVmodel_13/transformer_block_13/multi_head_attention_13/key/einsum/Einsum/ReadVariableOp2?
Nmodel_13/transformer_block_13/multi_head_attention_13/query/add/ReadVariableOpNmodel_13/transformer_block_13/multi_head_attention_13/query/add/ReadVariableOp2?
Xmodel_13/transformer_block_13/multi_head_attention_13/query/einsum/Einsum/ReadVariableOpXmodel_13/transformer_block_13/multi_head_attention_13/query/einsum/Einsum/ReadVariableOp2?
Nmodel_13/transformer_block_13/multi_head_attention_13/value/add/ReadVariableOpNmodel_13/transformer_block_13/multi_head_attention_13/value/add/ReadVariableOp2?
Xmodel_13/transformer_block_13/multi_head_attention_13/value/einsum/Einsum/ReadVariableOpXmodel_13/transformer_block_13/multi_head_attention_13/value/einsum/Einsum/ReadVariableOp2?
Kmodel_13/transformer_block_13/sequential_13/dense_52/BiasAdd/ReadVariableOpKmodel_13/transformer_block_13/sequential_13/dense_52/BiasAdd/ReadVariableOp2?
Mmodel_13/transformer_block_13/sequential_13/dense_52/Tensordot/ReadVariableOpMmodel_13/transformer_block_13/sequential_13/dense_52/Tensordot/ReadVariableOp2?
Kmodel_13/transformer_block_13/sequential_13/dense_53/BiasAdd/ReadVariableOpKmodel_13/transformer_block_13/sequential_13/dense_53/BiasAdd/ReadVariableOp2?
Mmodel_13/transformer_block_13/sequential_13/dense_53/Tensordot/ReadVariableOpMmodel_13/transformer_block_13/sequential_13/dense_53/Tensordot/ReadVariableOp:R N
(
_output_shapes
:??????????
"
_user_specified_name
input_14
?
?
-__inference_sequential_13_layer_call_fn_81245
dense_52_input
unknown:
??
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_52_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_13_layer_call_and_return_conditional_losses_81234u
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
-
_output_shapes
:???????????
(
_user_specified_namedense_52_input
?
r
V__inference_global_average_pooling1d_13_layer_call_and_return_conditional_losses_83524

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :h
MeanMeaninputsMean/reduction_indices:output:0*
T0*(
_output_shapes
:??????????V
IdentityIdentityMean:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?+
?	
C__inference_model_13_layer_call_and_return_conditional_losses_81671

inputs9
%token_and_position_embedding_13_81391:
??9
%token_and_position_embedding_13_81393:
?)?2
transformer_block_13_81582:??-
transformer_block_13_81584:	?2
transformer_block_13_81586:??-
transformer_block_13_81588:	?2
transformer_block_13_81590:??-
transformer_block_13_81592:	?2
transformer_block_13_81594:??)
transformer_block_13_81596:	?)
transformer_block_13_81598:	?)
transformer_block_13_81600:	?.
transformer_block_13_81602:
??)
transformer_block_13_81604:	?.
transformer_block_13_81606:
??)
transformer_block_13_81608:	?)
transformer_block_13_81610:	?)
transformer_block_13_81612:	?!
dense_54_81641:	?
dense_54_81643: 
dense_55_81665:
dense_55_81667:
identity?? dense_54/StatefulPartitionedCall? dense_55/StatefulPartitionedCall?7token_and_position_embedding_13/StatefulPartitionedCall?,transformer_block_13/StatefulPartitionedCall?
7token_and_position_embedding_13/StatefulPartitionedCallStatefulPartitionedCallinputs%token_and_position_embedding_13_81391%token_and_position_embedding_13_81393*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *c
f^R\
Z__inference_token_and_position_embedding_13_layer_call_and_return_conditional_losses_81390?
,transformer_block_13/StatefulPartitionedCallStatefulPartitionedCall@token_and_position_embedding_13/StatefulPartitionedCall:output:0transformer_block_13_81582transformer_block_13_81584transformer_block_13_81586transformer_block_13_81588transformer_block_13_81590transformer_block_13_81592transformer_block_13_81594transformer_block_13_81596transformer_block_13_81598transformer_block_13_81600transformer_block_13_81602transformer_block_13_81604transformer_block_13_81606transformer_block_13_81608transformer_block_13_81610transformer_block_13_81612*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_transformer_block_13_layer_call_and_return_conditional_losses_81581?
+global_average_pooling1d_13/PartitionedCallPartitionedCall5transformer_block_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *_
fZRX
V__inference_global_average_pooling1d_13_layer_call_and_return_conditional_losses_81620?
dropout_57/PartitionedCallPartitionedCall4global_average_pooling1d_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_57_layer_call_and_return_conditional_losses_81627?
 dense_54/StatefulPartitionedCallStatefulPartitionedCall#dropout_57/PartitionedCall:output:0dense_54_81641dense_54_81643*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_54_layer_call_and_return_conditional_losses_81640?
dropout_58/PartitionedCallPartitionedCall)dense_54/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_58_layer_call_and_return_conditional_losses_81651?
 dense_55/StatefulPartitionedCallStatefulPartitionedCall#dropout_58/PartitionedCall:output:0dense_55_81665dense_55_81667*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_55_layer_call_and_return_conditional_losses_81664x
IdentityIdentity)dense_55/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_54/StatefulPartitionedCall!^dense_55/StatefulPartitionedCall8^token_and_position_embedding_13/StatefulPartitionedCall-^transformer_block_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:??????????: : : : : : : : : : : : : : : : : : : : : : 2D
 dense_54/StatefulPartitionedCall dense_54/StatefulPartitionedCall2D
 dense_55/StatefulPartitionedCall dense_55/StatefulPartitionedCall2r
7token_and_position_embedding_13/StatefulPartitionedCall7token_and_position_embedding_13/StatefulPartitionedCall2\
,transformer_block_13/StatefulPartitionedCall,transformer_block_13/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
d
E__inference_dropout_58_layer_call_and_return_conditional_losses_81748

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
H__inference_sequential_13_layer_call_and_return_conditional_losses_81346
dense_52_input"
dense_52_81335:
??
dense_52_81337:	?"
dense_53_81340:
??
dense_53_81342:	?
identity?? dense_52/StatefulPartitionedCall? dense_53/StatefulPartitionedCall?
 dense_52/StatefulPartitionedCallStatefulPartitionedCalldense_52_inputdense_52_81335dense_52_81337*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_52_layer_call_and_return_conditional_losses_81191?
 dense_53/StatefulPartitionedCallStatefulPartitionedCall)dense_52/StatefulPartitionedCall:output:0dense_53_81340dense_53_81342*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_53_layer_call_and_return_conditional_losses_81227~
IdentityIdentity)dense_53/StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:????????????
NoOpNoOp!^dense_52/StatefulPartitionedCall!^dense_53/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : : : 2D
 dense_52/StatefulPartitionedCall dense_52/StatefulPartitionedCall2D
 dense_53/StatefulPartitionedCall dense_53/StatefulPartitionedCall:] Y
-
_output_shapes
:???????????
(
_user_specified_namedense_52_input
?+
?	
C__inference_model_13_layer_call_and_return_conditional_losses_82329
input_149
%token_and_position_embedding_13_82277:
??9
%token_and_position_embedding_13_82279:
?)?2
transformer_block_13_82282:??-
transformer_block_13_82284:	?2
transformer_block_13_82286:??-
transformer_block_13_82288:	?2
transformer_block_13_82290:??-
transformer_block_13_82292:	?2
transformer_block_13_82294:??)
transformer_block_13_82296:	?)
transformer_block_13_82298:	?)
transformer_block_13_82300:	?.
transformer_block_13_82302:
??)
transformer_block_13_82304:	?.
transformer_block_13_82306:
??)
transformer_block_13_82308:	?)
transformer_block_13_82310:	?)
transformer_block_13_82312:	?!
dense_54_82317:	?
dense_54_82319: 
dense_55_82323:
dense_55_82325:
identity?? dense_54/StatefulPartitionedCall? dense_55/StatefulPartitionedCall?7token_and_position_embedding_13/StatefulPartitionedCall?,transformer_block_13/StatefulPartitionedCall?
7token_and_position_embedding_13/StatefulPartitionedCallStatefulPartitionedCallinput_14%token_and_position_embedding_13_82277%token_and_position_embedding_13_82279*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *c
f^R\
Z__inference_token_and_position_embedding_13_layer_call_and_return_conditional_losses_81390?
,transformer_block_13/StatefulPartitionedCallStatefulPartitionedCall@token_and_position_embedding_13/StatefulPartitionedCall:output:0transformer_block_13_82282transformer_block_13_82284transformer_block_13_82286transformer_block_13_82288transformer_block_13_82290transformer_block_13_82292transformer_block_13_82294transformer_block_13_82296transformer_block_13_82298transformer_block_13_82300transformer_block_13_82302transformer_block_13_82304transformer_block_13_82306transformer_block_13_82308transformer_block_13_82310transformer_block_13_82312*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_transformer_block_13_layer_call_and_return_conditional_losses_81581?
+global_average_pooling1d_13/PartitionedCallPartitionedCall5transformer_block_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *_
fZRX
V__inference_global_average_pooling1d_13_layer_call_and_return_conditional_losses_81620?
dropout_57/PartitionedCallPartitionedCall4global_average_pooling1d_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_57_layer_call_and_return_conditional_losses_81627?
 dense_54/StatefulPartitionedCallStatefulPartitionedCall#dropout_57/PartitionedCall:output:0dense_54_82317dense_54_82319*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_54_layer_call_and_return_conditional_losses_81640?
dropout_58/PartitionedCallPartitionedCall)dense_54/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_58_layer_call_and_return_conditional_losses_81651?
 dense_55/StatefulPartitionedCallStatefulPartitionedCall#dropout_58/PartitionedCall:output:0dense_55_82323dense_55_82325*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_55_layer_call_and_return_conditional_losses_81664x
IdentityIdentity)dense_55/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_54/StatefulPartitionedCall!^dense_55/StatefulPartitionedCall8^token_and_position_embedding_13/StatefulPartitionedCall-^transformer_block_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:??????????: : : : : : : : : : : : : : : : : : : : : : 2D
 dense_54/StatefulPartitionedCall dense_54/StatefulPartitionedCall2D
 dense_55/StatefulPartitionedCall dense_55/StatefulPartitionedCall2r
7token_and_position_embedding_13/StatefulPartitionedCall7token_and_position_embedding_13/StatefulPartitionedCall2\
,transformer_block_13/StatefulPartitionedCall,transformer_block_13/StatefulPartitionedCall:R N
(
_output_shapes
:??????????
"
_user_specified_name
input_14
?
?
H__inference_sequential_13_layer_call_and_return_conditional_losses_81294

inputs"
dense_52_81283:
??
dense_52_81285:	?"
dense_53_81288:
??
dense_53_81290:	?
identity?? dense_52/StatefulPartitionedCall? dense_53/StatefulPartitionedCall?
 dense_52/StatefulPartitionedCallStatefulPartitionedCallinputsdense_52_81283dense_52_81285*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_52_layer_call_and_return_conditional_losses_81191?
 dense_53/StatefulPartitionedCallStatefulPartitionedCall)dense_52/StatefulPartitionedCall:output:0dense_53_81288dense_53_81290*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_53_layer_call_and_return_conditional_losses_81227~
IdentityIdentity)dense_53/StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:????????????
NoOpNoOp!^dense_52/StatefulPartitionedCall!^dense_53/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : : : 2D
 dense_52/StatefulPartitionedCall dense_52/StatefulPartitionedCall2D
 dense_53/StatefulPartitionedCall dense_53/StatefulPartitionedCall:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
(__inference_model_13_layer_call_fn_82490

inputs
unknown:
??
	unknown_0:
?)?!
	unknown_1:??
	unknown_2:	?!
	unknown_3:??
	unknown_4:	?!
	unknown_5:??
	unknown_6:	?!
	unknown_7:??
	unknown_8:	?
	unknown_9:	?

unknown_10:	?

unknown_11:
??

unknown_12:	?

unknown_13:
??

unknown_14:	?

unknown_15:	?

unknown_16:	?

unknown_17:	?

unknown_18:

unknown_19:

unknown_20:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_model_13_layer_call_and_return_conditional_losses_81671o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:??????????: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
F
*__inference_dropout_57_layer_call_fn_83529

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_57_layer_call_and_return_conditional_losses_81627a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
-__inference_sequential_13_layer_call_fn_83644

inputs
unknown:
??
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_13_layer_call_and_return_conditional_losses_81294u
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
r
V__inference_global_average_pooling1d_13_layer_call_and_return_conditional_losses_81620

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :h
MeanMeaninputsMean/reduction_indices:output:0*
T0*(
_output_shapes
:??????????V
IdentityIdentityMean:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?.
?

C__inference_model_13_layer_call_and_return_conditional_losses_82178

inputs9
%token_and_position_embedding_13_82126:
??9
%token_and_position_embedding_13_82128:
?)?2
transformer_block_13_82131:??-
transformer_block_13_82133:	?2
transformer_block_13_82135:??-
transformer_block_13_82137:	?2
transformer_block_13_82139:??-
transformer_block_13_82141:	?2
transformer_block_13_82143:??)
transformer_block_13_82145:	?)
transformer_block_13_82147:	?)
transformer_block_13_82149:	?.
transformer_block_13_82151:
??)
transformer_block_13_82153:	?.
transformer_block_13_82155:
??)
transformer_block_13_82157:	?)
transformer_block_13_82159:	?)
transformer_block_13_82161:	?!
dense_54_82166:	?
dense_54_82168: 
dense_55_82172:
dense_55_82174:
identity?? dense_54/StatefulPartitionedCall? dense_55/StatefulPartitionedCall?"dropout_57/StatefulPartitionedCall?"dropout_58/StatefulPartitionedCall?7token_and_position_embedding_13/StatefulPartitionedCall?,transformer_block_13/StatefulPartitionedCall?
7token_and_position_embedding_13/StatefulPartitionedCallStatefulPartitionedCallinputs%token_and_position_embedding_13_82126%token_and_position_embedding_13_82128*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *c
f^R\
Z__inference_token_and_position_embedding_13_layer_call_and_return_conditional_losses_81390?
,transformer_block_13/StatefulPartitionedCallStatefulPartitionedCall@token_and_position_embedding_13/StatefulPartitionedCall:output:0transformer_block_13_82131transformer_block_13_82133transformer_block_13_82135transformer_block_13_82137transformer_block_13_82139transformer_block_13_82141transformer_block_13_82143transformer_block_13_82145transformer_block_13_82147transformer_block_13_82149transformer_block_13_82151transformer_block_13_82153transformer_block_13_82155transformer_block_13_82157transformer_block_13_82159transformer_block_13_82161*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_transformer_block_13_layer_call_and_return_conditional_losses_82027?
+global_average_pooling1d_13/PartitionedCallPartitionedCall5transformer_block_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *_
fZRX
V__inference_global_average_pooling1d_13_layer_call_and_return_conditional_losses_81620?
"dropout_57/StatefulPartitionedCallStatefulPartitionedCall4global_average_pooling1d_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_57_layer_call_and_return_conditional_losses_81781?
 dense_54/StatefulPartitionedCallStatefulPartitionedCall+dropout_57/StatefulPartitionedCall:output:0dense_54_82166dense_54_82168*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_54_layer_call_and_return_conditional_losses_81640?
"dropout_58/StatefulPartitionedCallStatefulPartitionedCall)dense_54/StatefulPartitionedCall:output:0#^dropout_57/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_58_layer_call_and_return_conditional_losses_81748?
 dense_55/StatefulPartitionedCallStatefulPartitionedCall+dropout_58/StatefulPartitionedCall:output:0dense_55_82172dense_55_82174*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_55_layer_call_and_return_conditional_losses_81664x
IdentityIdentity)dense_55/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_54/StatefulPartitionedCall!^dense_55/StatefulPartitionedCall#^dropout_57/StatefulPartitionedCall#^dropout_58/StatefulPartitionedCall8^token_and_position_embedding_13/StatefulPartitionedCall-^transformer_block_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:??????????: : : : : : : : : : : : : : : : : : : : : : 2D
 dense_54/StatefulPartitionedCall dense_54/StatefulPartitionedCall2D
 dense_55/StatefulPartitionedCall dense_55/StatefulPartitionedCall2H
"dropout_57/StatefulPartitionedCall"dropout_57/StatefulPartitionedCall2H
"dropout_58/StatefulPartitionedCall"dropout_58/StatefulPartitionedCall2r
7token_and_position_embedding_13/StatefulPartitionedCall7token_and_position_embedding_13/StatefulPartitionedCall2\
,transformer_block_13/StatefulPartitionedCall,transformer_block_13/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
4__inference_transformer_block_13_layer_call_fn_83082

inputs
unknown:??
	unknown_0:	?!
	unknown_1:??
	unknown_2:	?!
	unknown_3:??
	unknown_4:	?!
	unknown_5:??
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:
??

unknown_10:	?

unknown_11:
??

unknown_12:	?

unknown_13:	?

unknown_14:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_transformer_block_13_layer_call_and_return_conditional_losses_81581u
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:???????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
W
;__inference_global_average_pooling1d_13_layer_call_fn_83507

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *_
fZRX
V__inference_global_average_pooling1d_13_layer_call_and_return_conditional_losses_81356i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
??
?
O__inference_transformer_block_13_layer_call_and_return_conditional_losses_83304

inputs[
Cmulti_head_attention_13_query_einsum_einsum_readvariableop_resource:??L
9multi_head_attention_13_query_add_readvariableop_resource:	?Y
Amulti_head_attention_13_key_einsum_einsum_readvariableop_resource:??J
7multi_head_attention_13_key_add_readvariableop_resource:	?[
Cmulti_head_attention_13_value_einsum_einsum_readvariableop_resource:??L
9multi_head_attention_13_value_add_readvariableop_resource:	?f
Nmulti_head_attention_13_attention_output_einsum_einsum_readvariableop_resource:??S
Dmulti_head_attention_13_attention_output_add_readvariableop_resource:	?C
4layer_normalization_26_mul_3_readvariableop_resource:	?A
2layer_normalization_26_add_readvariableop_resource:	?L
8sequential_13_dense_52_tensordot_readvariableop_resource:
??E
6sequential_13_dense_52_biasadd_readvariableop_resource:	?L
8sequential_13_dense_53_tensordot_readvariableop_resource:
??E
6sequential_13_dense_53_biasadd_readvariableop_resource:	?C
4layer_normalization_27_mul_3_readvariableop_resource:	?A
2layer_normalization_27_add_readvariableop_resource:	?
identity??)layer_normalization_26/add/ReadVariableOp?+layer_normalization_26/mul_3/ReadVariableOp?)layer_normalization_27/add/ReadVariableOp?+layer_normalization_27/mul_3/ReadVariableOp?;multi_head_attention_13/attention_output/add/ReadVariableOp?Emulti_head_attention_13/attention_output/einsum/Einsum/ReadVariableOp?.multi_head_attention_13/key/add/ReadVariableOp?8multi_head_attention_13/key/einsum/Einsum/ReadVariableOp?0multi_head_attention_13/query/add/ReadVariableOp?:multi_head_attention_13/query/einsum/Einsum/ReadVariableOp?0multi_head_attention_13/value/add/ReadVariableOp?:multi_head_attention_13/value/einsum/Einsum/ReadVariableOp?-sequential_13/dense_52/BiasAdd/ReadVariableOp?/sequential_13/dense_52/Tensordot/ReadVariableOp?-sequential_13/dense_53/BiasAdd/ReadVariableOp?/sequential_13/dense_53/Tensordot/ReadVariableOp?
:multi_head_attention_13/query/einsum/Einsum/ReadVariableOpReadVariableOpCmulti_head_attention_13_query_einsum_einsum_readvariableop_resource*$
_output_shapes
:??*
dtype0?
+multi_head_attention_13/query/einsum/EinsumEinsuminputsBmulti_head_attention_13/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*1
_output_shapes
:???????????*
equationabc,cde->abde?
0multi_head_attention_13/query/add/ReadVariableOpReadVariableOp9multi_head_attention_13_query_add_readvariableop_resource*
_output_shapes
:	?*
dtype0?
!multi_head_attention_13/query/addAddV24multi_head_attention_13/query/einsum/Einsum:output:08multi_head_attention_13/query/add/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
8multi_head_attention_13/key/einsum/Einsum/ReadVariableOpReadVariableOpAmulti_head_attention_13_key_einsum_einsum_readvariableop_resource*$
_output_shapes
:??*
dtype0?
)multi_head_attention_13/key/einsum/EinsumEinsuminputs@multi_head_attention_13/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*1
_output_shapes
:???????????*
equationabc,cde->abde?
.multi_head_attention_13/key/add/ReadVariableOpReadVariableOp7multi_head_attention_13_key_add_readvariableop_resource*
_output_shapes
:	?*
dtype0?
multi_head_attention_13/key/addAddV22multi_head_attention_13/key/einsum/Einsum:output:06multi_head_attention_13/key/add/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
:multi_head_attention_13/value/einsum/Einsum/ReadVariableOpReadVariableOpCmulti_head_attention_13_value_einsum_einsum_readvariableop_resource*$
_output_shapes
:??*
dtype0?
+multi_head_attention_13/value/einsum/EinsumEinsuminputsBmulti_head_attention_13/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*1
_output_shapes
:???????????*
equationabc,cde->abde?
0multi_head_attention_13/value/add/ReadVariableOpReadVariableOp9multi_head_attention_13_value_add_readvariableop_resource*
_output_shapes
:	?*
dtype0?
!multi_head_attention_13/value/addAddV24multi_head_attention_13/value/einsum/Einsum:output:08multi_head_attention_13/value/add/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????b
multi_head_attention_13/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *?А=?
multi_head_attention_13/MulMul%multi_head_attention_13/query/add:z:0&multi_head_attention_13/Mul/y:output:0*
T0*1
_output_shapes
:????????????
%multi_head_attention_13/einsum/EinsumEinsum#multi_head_attention_13/key/add:z:0multi_head_attention_13/Mul:z:0*
N*
T0*1
_output_shapes
:???????????*
equationaecd,abcd->acbe?
'multi_head_attention_13/softmax/SoftmaxSoftmax.multi_head_attention_13/einsum/Einsum:output:0*
T0*1
_output_shapes
:????????????
(multi_head_attention_13/dropout/IdentityIdentity1multi_head_attention_13/softmax/Softmax:softmax:0*
T0*1
_output_shapes
:????????????
'multi_head_attention_13/einsum_1/EinsumEinsum1multi_head_attention_13/dropout/Identity:output:0%multi_head_attention_13/value/add:z:0*
N*
T0*1
_output_shapes
:???????????*
equationacbe,aecd->abcd?
Emulti_head_attention_13/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpNmulti_head_attention_13_attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:??*
dtype0?
6multi_head_attention_13/attention_output/einsum/EinsumEinsum0multi_head_attention_13/einsum_1/Einsum:output:0Mmulti_head_attention_13/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*-
_output_shapes
:???????????*
equationabcd,cde->abe?
;multi_head_attention_13/attention_output/add/ReadVariableOpReadVariableOpDmulti_head_attention_13_attention_output_add_readvariableop_resource*
_output_shapes	
:?*
dtype0?
,multi_head_attention_13/attention_output/addAddV2?multi_head_attention_13/attention_output/einsum/Einsum:output:0Cmulti_head_attention_13/attention_output/add/ReadVariableOp:value:0*
T0*-
_output_shapes
:????????????
dropout_55/IdentityIdentity0multi_head_attention_13/attention_output/add:z:0*
T0*-
_output_shapes
:???????????j
addAddV2inputsdropout_55/Identity:output:0*
T0*-
_output_shapes
:???????????S
layer_normalization_26/ShapeShapeadd:z:0*
T0*
_output_shapes
:t
*layer_normalization_26/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,layer_normalization_26/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,layer_normalization_26/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$layer_normalization_26/strided_sliceStridedSlice%layer_normalization_26/Shape:output:03layer_normalization_26/strided_slice/stack:output:05layer_normalization_26/strided_slice/stack_1:output:05layer_normalization_26/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
layer_normalization_26/mul/xConst*
_output_shapes
: *
dtype0*
value	B :?
layer_normalization_26/mulMul%layer_normalization_26/mul/x:output:0-layer_normalization_26/strided_slice:output:0*
T0*
_output_shapes
: v
,layer_normalization_26/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_26/strided_slice_1StridedSlice%layer_normalization_26/Shape:output:05layer_normalization_26/strided_slice_1/stack:output:07layer_normalization_26/strided_slice_1/stack_1:output:07layer_normalization_26/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
layer_normalization_26/mul_1Mullayer_normalization_26/mul:z:0/layer_normalization_26/strided_slice_1:output:0*
T0*
_output_shapes
: v
,layer_normalization_26/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_26/strided_slice_2StridedSlice%layer_normalization_26/Shape:output:05layer_normalization_26/strided_slice_2/stack:output:07layer_normalization_26/strided_slice_2/stack_1:output:07layer_normalization_26/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
layer_normalization_26/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :?
layer_normalization_26/mul_2Mul'layer_normalization_26/mul_2/x:output:0/layer_normalization_26/strided_slice_2:output:0*
T0*
_output_shapes
: h
&layer_normalization_26/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :h
&layer_normalization_26/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
$layer_normalization_26/Reshape/shapePack/layer_normalization_26/Reshape/shape/0:output:0 layer_normalization_26/mul_1:z:0 layer_normalization_26/mul_2:z:0/layer_normalization_26/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
layer_normalization_26/ReshapeReshapeadd:z:0-layer_normalization_26/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????e
"layer_normalization_26/ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :??
 layer_normalization_26/ones/LessLess layer_normalization_26/mul_1:z:0+layer_normalization_26/ones/Less/y:output:0*
T0*
_output_shapes
: z
"layer_normalization_26/ones/packedPack layer_normalization_26/mul_1:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_26/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
layer_normalization_26/onesFill+layer_normalization_26/ones/packed:output:0*layer_normalization_26/ones/Const:output:0*
T0*#
_output_shapes
:?????????f
#layer_normalization_26/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :??
!layer_normalization_26/zeros/LessLess layer_normalization_26/mul_1:z:0,layer_normalization_26/zeros/Less/y:output:0*
T0*
_output_shapes
: {
#layer_normalization_26/zeros/packedPack layer_normalization_26/mul_1:z:0*
N*
T0*
_output_shapes
:g
"layer_normalization_26/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
layer_normalization_26/zerosFill,layer_normalization_26/zeros/packed:output:0+layer_normalization_26/zeros/Const:output:0*
T0*#
_output_shapes
:?????????_
layer_normalization_26/ConstConst*
_output_shapes
: *
dtype0*
valueB a
layer_normalization_26/Const_1Const*
_output_shapes
: *
dtype0*
valueB ?
'layer_normalization_26/FusedBatchNormV3FusedBatchNormV3'layer_normalization_26/Reshape:output:0$layer_normalization_26/ones:output:0%layer_normalization_26/zeros:output:0%layer_normalization_26/Const:output:0'layer_normalization_26/Const_1:output:0*
T0*
U0*p
_output_shapes^
\:??????????:?????????:?????????:?????????:?????????:*
data_formatNCHW?
 layer_normalization_26/Reshape_1Reshape+layer_normalization_26/FusedBatchNormV3:y:0%layer_normalization_26/Shape:output:0*
T0*-
_output_shapes
:????????????
+layer_normalization_26/mul_3/ReadVariableOpReadVariableOp4layer_normalization_26_mul_3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
layer_normalization_26/mul_3Mul)layer_normalization_26/Reshape_1:output:03layer_normalization_26/mul_3/ReadVariableOp:value:0*
T0*-
_output_shapes
:????????????
)layer_normalization_26/add/ReadVariableOpReadVariableOp2layer_normalization_26_add_readvariableop_resource*
_output_shapes	
:?*
dtype0?
layer_normalization_26/addAddV2 layer_normalization_26/mul_3:z:01layer_normalization_26/add/ReadVariableOp:value:0*
T0*-
_output_shapes
:????????????
/sequential_13/dense_52/Tensordot/ReadVariableOpReadVariableOp8sequential_13_dense_52_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype0o
%sequential_13/dense_52/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:v
%sequential_13/dense_52/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       t
&sequential_13/dense_52/Tensordot/ShapeShapelayer_normalization_26/add:z:0*
T0*
_output_shapes
:p
.sequential_13/dense_52/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
)sequential_13/dense_52/Tensordot/GatherV2GatherV2/sequential_13/dense_52/Tensordot/Shape:output:0.sequential_13/dense_52/Tensordot/free:output:07sequential_13/dense_52/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:r
0sequential_13/dense_52/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+sequential_13/dense_52/Tensordot/GatherV2_1GatherV2/sequential_13/dense_52/Tensordot/Shape:output:0.sequential_13/dense_52/Tensordot/axes:output:09sequential_13/dense_52/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
&sequential_13/dense_52/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
%sequential_13/dense_52/Tensordot/ProdProd2sequential_13/dense_52/Tensordot/GatherV2:output:0/sequential_13/dense_52/Tensordot/Const:output:0*
T0*
_output_shapes
: r
(sequential_13/dense_52/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
'sequential_13/dense_52/Tensordot/Prod_1Prod4sequential_13/dense_52/Tensordot/GatherV2_1:output:01sequential_13/dense_52/Tensordot/Const_1:output:0*
T0*
_output_shapes
: n
,sequential_13/dense_52/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
'sequential_13/dense_52/Tensordot/concatConcatV2.sequential_13/dense_52/Tensordot/free:output:0.sequential_13/dense_52/Tensordot/axes:output:05sequential_13/dense_52/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
&sequential_13/dense_52/Tensordot/stackPack.sequential_13/dense_52/Tensordot/Prod:output:00sequential_13/dense_52/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
*sequential_13/dense_52/Tensordot/transpose	Transposelayer_normalization_26/add:z:00sequential_13/dense_52/Tensordot/concat:output:0*
T0*-
_output_shapes
:????????????
(sequential_13/dense_52/Tensordot/ReshapeReshape.sequential_13/dense_52/Tensordot/transpose:y:0/sequential_13/dense_52/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
'sequential_13/dense_52/Tensordot/MatMulMatMul1sequential_13/dense_52/Tensordot/Reshape:output:07sequential_13/dense_52/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
(sequential_13/dense_52/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?p
.sequential_13/dense_52/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
)sequential_13/dense_52/Tensordot/concat_1ConcatV22sequential_13/dense_52/Tensordot/GatherV2:output:01sequential_13/dense_52/Tensordot/Const_2:output:07sequential_13/dense_52/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
 sequential_13/dense_52/TensordotReshape1sequential_13/dense_52/Tensordot/MatMul:product:02sequential_13/dense_52/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:????????????
-sequential_13/dense_52/BiasAdd/ReadVariableOpReadVariableOp6sequential_13_dense_52_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_13/dense_52/BiasAddBiasAdd)sequential_13/dense_52/Tensordot:output:05sequential_13/dense_52/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:????????????
sequential_13/dense_52/ReluRelu'sequential_13/dense_52/BiasAdd:output:0*
T0*-
_output_shapes
:????????????
/sequential_13/dense_53/Tensordot/ReadVariableOpReadVariableOp8sequential_13_dense_53_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype0o
%sequential_13/dense_53/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:v
%sequential_13/dense_53/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
&sequential_13/dense_53/Tensordot/ShapeShape)sequential_13/dense_52/Relu:activations:0*
T0*
_output_shapes
:p
.sequential_13/dense_53/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
)sequential_13/dense_53/Tensordot/GatherV2GatherV2/sequential_13/dense_53/Tensordot/Shape:output:0.sequential_13/dense_53/Tensordot/free:output:07sequential_13/dense_53/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:r
0sequential_13/dense_53/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+sequential_13/dense_53/Tensordot/GatherV2_1GatherV2/sequential_13/dense_53/Tensordot/Shape:output:0.sequential_13/dense_53/Tensordot/axes:output:09sequential_13/dense_53/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
&sequential_13/dense_53/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
%sequential_13/dense_53/Tensordot/ProdProd2sequential_13/dense_53/Tensordot/GatherV2:output:0/sequential_13/dense_53/Tensordot/Const:output:0*
T0*
_output_shapes
: r
(sequential_13/dense_53/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
'sequential_13/dense_53/Tensordot/Prod_1Prod4sequential_13/dense_53/Tensordot/GatherV2_1:output:01sequential_13/dense_53/Tensordot/Const_1:output:0*
T0*
_output_shapes
: n
,sequential_13/dense_53/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
'sequential_13/dense_53/Tensordot/concatConcatV2.sequential_13/dense_53/Tensordot/free:output:0.sequential_13/dense_53/Tensordot/axes:output:05sequential_13/dense_53/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
&sequential_13/dense_53/Tensordot/stackPack.sequential_13/dense_53/Tensordot/Prod:output:00sequential_13/dense_53/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
*sequential_13/dense_53/Tensordot/transpose	Transpose)sequential_13/dense_52/Relu:activations:00sequential_13/dense_53/Tensordot/concat:output:0*
T0*-
_output_shapes
:????????????
(sequential_13/dense_53/Tensordot/ReshapeReshape.sequential_13/dense_53/Tensordot/transpose:y:0/sequential_13/dense_53/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
'sequential_13/dense_53/Tensordot/MatMulMatMul1sequential_13/dense_53/Tensordot/Reshape:output:07sequential_13/dense_53/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
(sequential_13/dense_53/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?p
.sequential_13/dense_53/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
)sequential_13/dense_53/Tensordot/concat_1ConcatV22sequential_13/dense_53/Tensordot/GatherV2:output:01sequential_13/dense_53/Tensordot/Const_2:output:07sequential_13/dense_53/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
 sequential_13/dense_53/TensordotReshape1sequential_13/dense_53/Tensordot/MatMul:product:02sequential_13/dense_53/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:????????????
-sequential_13/dense_53/BiasAdd/ReadVariableOpReadVariableOp6sequential_13_dense_53_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_13/dense_53/BiasAddBiasAdd)sequential_13/dense_53/Tensordot:output:05sequential_13/dense_53/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:????????????
dropout_56/IdentityIdentity'sequential_13/dense_53/BiasAdd:output:0*
T0*-
_output_shapes
:????????????
add_1AddV2layer_normalization_26/add:z:0dropout_56/Identity:output:0*
T0*-
_output_shapes
:???????????U
layer_normalization_27/ShapeShape	add_1:z:0*
T0*
_output_shapes
:t
*layer_normalization_27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,layer_normalization_27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,layer_normalization_27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$layer_normalization_27/strided_sliceStridedSlice%layer_normalization_27/Shape:output:03layer_normalization_27/strided_slice/stack:output:05layer_normalization_27/strided_slice/stack_1:output:05layer_normalization_27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
layer_normalization_27/mul/xConst*
_output_shapes
: *
dtype0*
value	B :?
layer_normalization_27/mulMul%layer_normalization_27/mul/x:output:0-layer_normalization_27/strided_slice:output:0*
T0*
_output_shapes
: v
,layer_normalization_27/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_27/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_27/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_27/strided_slice_1StridedSlice%layer_normalization_27/Shape:output:05layer_normalization_27/strided_slice_1/stack:output:07layer_normalization_27/strided_slice_1/stack_1:output:07layer_normalization_27/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
layer_normalization_27/mul_1Mullayer_normalization_27/mul:z:0/layer_normalization_27/strided_slice_1:output:0*
T0*
_output_shapes
: v
,layer_normalization_27/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_27/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_27/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_27/strided_slice_2StridedSlice%layer_normalization_27/Shape:output:05layer_normalization_27/strided_slice_2/stack:output:07layer_normalization_27/strided_slice_2/stack_1:output:07layer_normalization_27/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
layer_normalization_27/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :?
layer_normalization_27/mul_2Mul'layer_normalization_27/mul_2/x:output:0/layer_normalization_27/strided_slice_2:output:0*
T0*
_output_shapes
: h
&layer_normalization_27/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :h
&layer_normalization_27/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
$layer_normalization_27/Reshape/shapePack/layer_normalization_27/Reshape/shape/0:output:0 layer_normalization_27/mul_1:z:0 layer_normalization_27/mul_2:z:0/layer_normalization_27/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
layer_normalization_27/ReshapeReshape	add_1:z:0-layer_normalization_27/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????e
"layer_normalization_27/ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :??
 layer_normalization_27/ones/LessLess layer_normalization_27/mul_1:z:0+layer_normalization_27/ones/Less/y:output:0*
T0*
_output_shapes
: z
"layer_normalization_27/ones/packedPack layer_normalization_27/mul_1:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_27/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
layer_normalization_27/onesFill+layer_normalization_27/ones/packed:output:0*layer_normalization_27/ones/Const:output:0*
T0*#
_output_shapes
:?????????f
#layer_normalization_27/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :??
!layer_normalization_27/zeros/LessLess layer_normalization_27/mul_1:z:0,layer_normalization_27/zeros/Less/y:output:0*
T0*
_output_shapes
: {
#layer_normalization_27/zeros/packedPack layer_normalization_27/mul_1:z:0*
N*
T0*
_output_shapes
:g
"layer_normalization_27/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
layer_normalization_27/zerosFill,layer_normalization_27/zeros/packed:output:0+layer_normalization_27/zeros/Const:output:0*
T0*#
_output_shapes
:?????????_
layer_normalization_27/ConstConst*
_output_shapes
: *
dtype0*
valueB a
layer_normalization_27/Const_1Const*
_output_shapes
: *
dtype0*
valueB ?
'layer_normalization_27/FusedBatchNormV3FusedBatchNormV3'layer_normalization_27/Reshape:output:0$layer_normalization_27/ones:output:0%layer_normalization_27/zeros:output:0%layer_normalization_27/Const:output:0'layer_normalization_27/Const_1:output:0*
T0*
U0*p
_output_shapes^
\:??????????:?????????:?????????:?????????:?????????:*
data_formatNCHW?
 layer_normalization_27/Reshape_1Reshape+layer_normalization_27/FusedBatchNormV3:y:0%layer_normalization_27/Shape:output:0*
T0*-
_output_shapes
:????????????
+layer_normalization_27/mul_3/ReadVariableOpReadVariableOp4layer_normalization_27_mul_3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
layer_normalization_27/mul_3Mul)layer_normalization_27/Reshape_1:output:03layer_normalization_27/mul_3/ReadVariableOp:value:0*
T0*-
_output_shapes
:????????????
)layer_normalization_27/add/ReadVariableOpReadVariableOp2layer_normalization_27_add_readvariableop_resource*
_output_shapes	
:?*
dtype0?
layer_normalization_27/addAddV2 layer_normalization_27/mul_3:z:01layer_normalization_27/add/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????s
IdentityIdentitylayer_normalization_27/add:z:0^NoOp*
T0*-
_output_shapes
:????????????
NoOpNoOp*^layer_normalization_26/add/ReadVariableOp,^layer_normalization_26/mul_3/ReadVariableOp*^layer_normalization_27/add/ReadVariableOp,^layer_normalization_27/mul_3/ReadVariableOp<^multi_head_attention_13/attention_output/add/ReadVariableOpF^multi_head_attention_13/attention_output/einsum/Einsum/ReadVariableOp/^multi_head_attention_13/key/add/ReadVariableOp9^multi_head_attention_13/key/einsum/Einsum/ReadVariableOp1^multi_head_attention_13/query/add/ReadVariableOp;^multi_head_attention_13/query/einsum/Einsum/ReadVariableOp1^multi_head_attention_13/value/add/ReadVariableOp;^multi_head_attention_13/value/einsum/Einsum/ReadVariableOp.^sequential_13/dense_52/BiasAdd/ReadVariableOp0^sequential_13/dense_52/Tensordot/ReadVariableOp.^sequential_13/dense_53/BiasAdd/ReadVariableOp0^sequential_13/dense_53/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:???????????: : : : : : : : : : : : : : : : 2V
)layer_normalization_26/add/ReadVariableOp)layer_normalization_26/add/ReadVariableOp2Z
+layer_normalization_26/mul_3/ReadVariableOp+layer_normalization_26/mul_3/ReadVariableOp2V
)layer_normalization_27/add/ReadVariableOp)layer_normalization_27/add/ReadVariableOp2Z
+layer_normalization_27/mul_3/ReadVariableOp+layer_normalization_27/mul_3/ReadVariableOp2z
;multi_head_attention_13/attention_output/add/ReadVariableOp;multi_head_attention_13/attention_output/add/ReadVariableOp2?
Emulti_head_attention_13/attention_output/einsum/Einsum/ReadVariableOpEmulti_head_attention_13/attention_output/einsum/Einsum/ReadVariableOp2`
.multi_head_attention_13/key/add/ReadVariableOp.multi_head_attention_13/key/add/ReadVariableOp2t
8multi_head_attention_13/key/einsum/Einsum/ReadVariableOp8multi_head_attention_13/key/einsum/Einsum/ReadVariableOp2d
0multi_head_attention_13/query/add/ReadVariableOp0multi_head_attention_13/query/add/ReadVariableOp2x
:multi_head_attention_13/query/einsum/Einsum/ReadVariableOp:multi_head_attention_13/query/einsum/Einsum/ReadVariableOp2d
0multi_head_attention_13/value/add/ReadVariableOp0multi_head_attention_13/value/add/ReadVariableOp2x
:multi_head_attention_13/value/einsum/Einsum/ReadVariableOp:multi_head_attention_13/value/einsum/Einsum/ReadVariableOp2^
-sequential_13/dense_52/BiasAdd/ReadVariableOp-sequential_13/dense_52/BiasAdd/ReadVariableOp2b
/sequential_13/dense_52/Tensordot/ReadVariableOp/sequential_13/dense_52/Tensordot/ReadVariableOp2^
-sequential_13/dense_53/BiasAdd/ReadVariableOp-sequential_13/dense_53/BiasAdd/ReadVariableOp2b
/sequential_13/dense_53/Tensordot/ReadVariableOp/sequential_13/dense_53/Tensordot/ReadVariableOp:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
(__inference_dense_52_layer_call_fn_83767

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_52_layer_call_and_return_conditional_losses_81191u
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_82441
input_14
unknown:
??
	unknown_0:
?)?!
	unknown_1:??
	unknown_2:	?!
	unknown_3:??
	unknown_4:	?!
	unknown_5:??
	unknown_6:	?!
	unknown_7:??
	unknown_8:	?
	unknown_9:	?

unknown_10:	?

unknown_11:
??

unknown_12:	?

unknown_13:
??

unknown_14:	?

unknown_15:	?

unknown_16:	?

unknown_17:	?

unknown_18:

unknown_19:

unknown_20:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_14unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_81153o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:??????????: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:??????????
"
_user_specified_name
input_14
??
?+
__inference__traced_save_84085
file_prefix.
*savev2_dense_54_kernel_read_readvariableop,
(savev2_dense_54_bias_read_readvariableop.
*savev2_dense_55_kernel_read_readvariableop,
(savev2_dense_55_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopV
Rsavev2_token_and_position_embedding_13_embedding_26_embeddings_read_readvariableopV
Rsavev2_token_and_position_embedding_13_embedding_27_embeddings_read_readvariableopX
Tsavev2_transformer_block_13_multi_head_attention_13_query_kernel_read_readvariableopV
Rsavev2_transformer_block_13_multi_head_attention_13_query_bias_read_readvariableopV
Rsavev2_transformer_block_13_multi_head_attention_13_key_kernel_read_readvariableopT
Psavev2_transformer_block_13_multi_head_attention_13_key_bias_read_readvariableopX
Tsavev2_transformer_block_13_multi_head_attention_13_value_kernel_read_readvariableopV
Rsavev2_transformer_block_13_multi_head_attention_13_value_bias_read_readvariableopc
_savev2_transformer_block_13_multi_head_attention_13_attention_output_kernel_read_readvariableopa
]savev2_transformer_block_13_multi_head_attention_13_attention_output_bias_read_readvariableop.
*savev2_dense_52_kernel_read_readvariableop,
(savev2_dense_52_bias_read_readvariableop.
*savev2_dense_53_kernel_read_readvariableop,
(savev2_dense_53_bias_read_readvariableopP
Lsavev2_transformer_block_13_layer_normalization_26_gamma_read_readvariableopO
Ksavev2_transformer_block_13_layer_normalization_26_beta_read_readvariableopP
Lsavev2_transformer_block_13_layer_normalization_27_gamma_read_readvariableopO
Ksavev2_transformer_block_13_layer_normalization_27_beta_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_54_kernel_m_read_readvariableop3
/savev2_adam_dense_54_bias_m_read_readvariableop5
1savev2_adam_dense_55_kernel_m_read_readvariableop3
/savev2_adam_dense_55_bias_m_read_readvariableop]
Ysavev2_adam_token_and_position_embedding_13_embedding_26_embeddings_m_read_readvariableop]
Ysavev2_adam_token_and_position_embedding_13_embedding_27_embeddings_m_read_readvariableop_
[savev2_adam_transformer_block_13_multi_head_attention_13_query_kernel_m_read_readvariableop]
Ysavev2_adam_transformer_block_13_multi_head_attention_13_query_bias_m_read_readvariableop]
Ysavev2_adam_transformer_block_13_multi_head_attention_13_key_kernel_m_read_readvariableop[
Wsavev2_adam_transformer_block_13_multi_head_attention_13_key_bias_m_read_readvariableop_
[savev2_adam_transformer_block_13_multi_head_attention_13_value_kernel_m_read_readvariableop]
Ysavev2_adam_transformer_block_13_multi_head_attention_13_value_bias_m_read_readvariableopj
fsavev2_adam_transformer_block_13_multi_head_attention_13_attention_output_kernel_m_read_readvariableoph
dsavev2_adam_transformer_block_13_multi_head_attention_13_attention_output_bias_m_read_readvariableop5
1savev2_adam_dense_52_kernel_m_read_readvariableop3
/savev2_adam_dense_52_bias_m_read_readvariableop5
1savev2_adam_dense_53_kernel_m_read_readvariableop3
/savev2_adam_dense_53_bias_m_read_readvariableopW
Ssavev2_adam_transformer_block_13_layer_normalization_26_gamma_m_read_readvariableopV
Rsavev2_adam_transformer_block_13_layer_normalization_26_beta_m_read_readvariableopW
Ssavev2_adam_transformer_block_13_layer_normalization_27_gamma_m_read_readvariableopV
Rsavev2_adam_transformer_block_13_layer_normalization_27_beta_m_read_readvariableop5
1savev2_adam_dense_54_kernel_v_read_readvariableop3
/savev2_adam_dense_54_bias_v_read_readvariableop5
1savev2_adam_dense_55_kernel_v_read_readvariableop3
/savev2_adam_dense_55_bias_v_read_readvariableop]
Ysavev2_adam_token_and_position_embedding_13_embedding_26_embeddings_v_read_readvariableop]
Ysavev2_adam_token_and_position_embedding_13_embedding_27_embeddings_v_read_readvariableop_
[savev2_adam_transformer_block_13_multi_head_attention_13_query_kernel_v_read_readvariableop]
Ysavev2_adam_transformer_block_13_multi_head_attention_13_query_bias_v_read_readvariableop]
Ysavev2_adam_transformer_block_13_multi_head_attention_13_key_kernel_v_read_readvariableop[
Wsavev2_adam_transformer_block_13_multi_head_attention_13_key_bias_v_read_readvariableop_
[savev2_adam_transformer_block_13_multi_head_attention_13_value_kernel_v_read_readvariableop]
Ysavev2_adam_transformer_block_13_multi_head_attention_13_value_bias_v_read_readvariableopj
fsavev2_adam_transformer_block_13_multi_head_attention_13_attention_output_kernel_v_read_readvariableoph
dsavev2_adam_transformer_block_13_multi_head_attention_13_attention_output_bias_v_read_readvariableop5
1savev2_adam_dense_52_kernel_v_read_readvariableop3
/savev2_adam_dense_52_bias_v_read_readvariableop5
1savev2_adam_dense_53_kernel_v_read_readvariableop3
/savev2_adam_dense_53_bias_v_read_readvariableopW
Ssavev2_adam_transformer_block_13_layer_normalization_26_gamma_v_read_readvariableopV
Rsavev2_adam_transformer_block_13_layer_normalization_26_beta_v_read_readvariableopW
Ssavev2_adam_transformer_block_13_layer_normalization_27_gamma_v_read_readvariableopV
Rsavev2_adam_transformer_block_13_layer_normalization_27_beta_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?$
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*?#
value?#B?#LB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*?
value?B?LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?*
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_54_kernel_read_readvariableop(savev2_dense_54_bias_read_readvariableop*savev2_dense_55_kernel_read_readvariableop(savev2_dense_55_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopRsavev2_token_and_position_embedding_13_embedding_26_embeddings_read_readvariableopRsavev2_token_and_position_embedding_13_embedding_27_embeddings_read_readvariableopTsavev2_transformer_block_13_multi_head_attention_13_query_kernel_read_readvariableopRsavev2_transformer_block_13_multi_head_attention_13_query_bias_read_readvariableopRsavev2_transformer_block_13_multi_head_attention_13_key_kernel_read_readvariableopPsavev2_transformer_block_13_multi_head_attention_13_key_bias_read_readvariableopTsavev2_transformer_block_13_multi_head_attention_13_value_kernel_read_readvariableopRsavev2_transformer_block_13_multi_head_attention_13_value_bias_read_readvariableop_savev2_transformer_block_13_multi_head_attention_13_attention_output_kernel_read_readvariableop]savev2_transformer_block_13_multi_head_attention_13_attention_output_bias_read_readvariableop*savev2_dense_52_kernel_read_readvariableop(savev2_dense_52_bias_read_readvariableop*savev2_dense_53_kernel_read_readvariableop(savev2_dense_53_bias_read_readvariableopLsavev2_transformer_block_13_layer_normalization_26_gamma_read_readvariableopKsavev2_transformer_block_13_layer_normalization_26_beta_read_readvariableopLsavev2_transformer_block_13_layer_normalization_27_gamma_read_readvariableopKsavev2_transformer_block_13_layer_normalization_27_beta_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_54_kernel_m_read_readvariableop/savev2_adam_dense_54_bias_m_read_readvariableop1savev2_adam_dense_55_kernel_m_read_readvariableop/savev2_adam_dense_55_bias_m_read_readvariableopYsavev2_adam_token_and_position_embedding_13_embedding_26_embeddings_m_read_readvariableopYsavev2_adam_token_and_position_embedding_13_embedding_27_embeddings_m_read_readvariableop[savev2_adam_transformer_block_13_multi_head_attention_13_query_kernel_m_read_readvariableopYsavev2_adam_transformer_block_13_multi_head_attention_13_query_bias_m_read_readvariableopYsavev2_adam_transformer_block_13_multi_head_attention_13_key_kernel_m_read_readvariableopWsavev2_adam_transformer_block_13_multi_head_attention_13_key_bias_m_read_readvariableop[savev2_adam_transformer_block_13_multi_head_attention_13_value_kernel_m_read_readvariableopYsavev2_adam_transformer_block_13_multi_head_attention_13_value_bias_m_read_readvariableopfsavev2_adam_transformer_block_13_multi_head_attention_13_attention_output_kernel_m_read_readvariableopdsavev2_adam_transformer_block_13_multi_head_attention_13_attention_output_bias_m_read_readvariableop1savev2_adam_dense_52_kernel_m_read_readvariableop/savev2_adam_dense_52_bias_m_read_readvariableop1savev2_adam_dense_53_kernel_m_read_readvariableop/savev2_adam_dense_53_bias_m_read_readvariableopSsavev2_adam_transformer_block_13_layer_normalization_26_gamma_m_read_readvariableopRsavev2_adam_transformer_block_13_layer_normalization_26_beta_m_read_readvariableopSsavev2_adam_transformer_block_13_layer_normalization_27_gamma_m_read_readvariableopRsavev2_adam_transformer_block_13_layer_normalization_27_beta_m_read_readvariableop1savev2_adam_dense_54_kernel_v_read_readvariableop/savev2_adam_dense_54_bias_v_read_readvariableop1savev2_adam_dense_55_kernel_v_read_readvariableop/savev2_adam_dense_55_bias_v_read_readvariableopYsavev2_adam_token_and_position_embedding_13_embedding_26_embeddings_v_read_readvariableopYsavev2_adam_token_and_position_embedding_13_embedding_27_embeddings_v_read_readvariableop[savev2_adam_transformer_block_13_multi_head_attention_13_query_kernel_v_read_readvariableopYsavev2_adam_transformer_block_13_multi_head_attention_13_query_bias_v_read_readvariableopYsavev2_adam_transformer_block_13_multi_head_attention_13_key_kernel_v_read_readvariableopWsavev2_adam_transformer_block_13_multi_head_attention_13_key_bias_v_read_readvariableop[savev2_adam_transformer_block_13_multi_head_attention_13_value_kernel_v_read_readvariableopYsavev2_adam_transformer_block_13_multi_head_attention_13_value_bias_v_read_readvariableopfsavev2_adam_transformer_block_13_multi_head_attention_13_attention_output_kernel_v_read_readvariableopdsavev2_adam_transformer_block_13_multi_head_attention_13_attention_output_bias_v_read_readvariableop1savev2_adam_dense_52_kernel_v_read_readvariableop/savev2_adam_dense_52_bias_v_read_readvariableop1savev2_adam_dense_53_kernel_v_read_readvariableop/savev2_adam_dense_53_bias_v_read_readvariableopSsavev2_adam_transformer_block_13_layer_normalization_26_gamma_v_read_readvariableopRsavev2_adam_transformer_block_13_layer_normalization_26_beta_v_read_readvariableopSsavev2_adam_transformer_block_13_layer_normalization_27_gamma_v_read_readvariableopRsavev2_adam_transformer_block_13_layer_normalization_27_beta_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *Z
dtypesP
N2L	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :	?:::: : : : : :
?)?:
??:??:	?:??:	?:??:	?:??:?:
??:?:
??:?:?:?:?:?: : : : :	?::::
?)?:
??:??:	?:??:	?:??:	?:??:?:
??:?:
??:?:?:?:?:?:	?::::
?)?:
??:??:	?:??:	?:??:	?:??:?:
??:?:
??:?:?:?:?:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :&
"
 
_output_shapes
:
?)?:&"
 
_output_shapes
:
??:*&
$
_output_shapes
:??:%!

_output_shapes
:	?:*&
$
_output_shapes
:??:%!

_output_shapes
:	?:*&
$
_output_shapes
:??:%!

_output_shapes
:	?:*&
$
_output_shapes
:??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :% !

_output_shapes
:	?: !

_output_shapes
::$" 

_output_shapes

:: #

_output_shapes
::&$"
 
_output_shapes
:
?)?:&%"
 
_output_shapes
:
??:*&&
$
_output_shapes
:??:%'!

_output_shapes
:	?:*(&
$
_output_shapes
:??:%)!

_output_shapes
:	?:**&
$
_output_shapes
:??:%+!

_output_shapes
:	?:*,&
$
_output_shapes
:??:!-

_output_shapes	
:?:&."
 
_output_shapes
:
??:!/

_output_shapes	
:?:&0"
 
_output_shapes
:
??:!1

_output_shapes	
:?:!2

_output_shapes	
:?:!3

_output_shapes	
:?:!4

_output_shapes	
:?:!5

_output_shapes	
:?:%6!

_output_shapes
:	?: 7

_output_shapes
::$8 

_output_shapes

:: 9

_output_shapes
::&:"
 
_output_shapes
:
?)?:&;"
 
_output_shapes
:
??:*<&
$
_output_shapes
:??:%=!

_output_shapes
:	?:*>&
$
_output_shapes
:??:%?!

_output_shapes
:	?:*@&
$
_output_shapes
:??:%A!

_output_shapes
:	?:*B&
$
_output_shapes
:??:!C

_output_shapes	
:?:&D"
 
_output_shapes
:
??:!E

_output_shapes	
:?:&F"
 
_output_shapes
:
??:!G

_output_shapes	
:?:!H

_output_shapes	
:?:!I

_output_shapes	
:?:!J

_output_shapes	
:?:!K

_output_shapes	
:?:L

_output_shapes
: 
??
?
C__inference_model_13_layer_call_and_return_conditional_losses_82762

inputsW
Ctoken_and_position_embedding_13_embedding_27_embedding_lookup_82550:
??W
Ctoken_and_position_embedding_13_embedding_26_embedding_lookup_82556:
?)?p
Xtransformer_block_13_multi_head_attention_13_query_einsum_einsum_readvariableop_resource:??a
Ntransformer_block_13_multi_head_attention_13_query_add_readvariableop_resource:	?n
Vtransformer_block_13_multi_head_attention_13_key_einsum_einsum_readvariableop_resource:??_
Ltransformer_block_13_multi_head_attention_13_key_add_readvariableop_resource:	?p
Xtransformer_block_13_multi_head_attention_13_value_einsum_einsum_readvariableop_resource:??a
Ntransformer_block_13_multi_head_attention_13_value_add_readvariableop_resource:	?{
ctransformer_block_13_multi_head_attention_13_attention_output_einsum_einsum_readvariableop_resource:??h
Ytransformer_block_13_multi_head_attention_13_attention_output_add_readvariableop_resource:	?X
Itransformer_block_13_layer_normalization_26_mul_3_readvariableop_resource:	?V
Gtransformer_block_13_layer_normalization_26_add_readvariableop_resource:	?a
Mtransformer_block_13_sequential_13_dense_52_tensordot_readvariableop_resource:
??Z
Ktransformer_block_13_sequential_13_dense_52_biasadd_readvariableop_resource:	?a
Mtransformer_block_13_sequential_13_dense_53_tensordot_readvariableop_resource:
??Z
Ktransformer_block_13_sequential_13_dense_53_biasadd_readvariableop_resource:	?X
Itransformer_block_13_layer_normalization_27_mul_3_readvariableop_resource:	?V
Gtransformer_block_13_layer_normalization_27_add_readvariableop_resource:	?:
'dense_54_matmul_readvariableop_resource:	?6
(dense_54_biasadd_readvariableop_resource:9
'dense_55_matmul_readvariableop_resource:6
(dense_55_biasadd_readvariableop_resource:
identity??dense_54/BiasAdd/ReadVariableOp?dense_54/MatMul/ReadVariableOp?dense_55/BiasAdd/ReadVariableOp?dense_55/MatMul/ReadVariableOp?=token_and_position_embedding_13/embedding_26/embedding_lookup?=token_and_position_embedding_13/embedding_27/embedding_lookup?>transformer_block_13/layer_normalization_26/add/ReadVariableOp?@transformer_block_13/layer_normalization_26/mul_3/ReadVariableOp?>transformer_block_13/layer_normalization_27/add/ReadVariableOp?@transformer_block_13/layer_normalization_27/mul_3/ReadVariableOp?Ptransformer_block_13/multi_head_attention_13/attention_output/add/ReadVariableOp?Ztransformer_block_13/multi_head_attention_13/attention_output/einsum/Einsum/ReadVariableOp?Ctransformer_block_13/multi_head_attention_13/key/add/ReadVariableOp?Mtransformer_block_13/multi_head_attention_13/key/einsum/Einsum/ReadVariableOp?Etransformer_block_13/multi_head_attention_13/query/add/ReadVariableOp?Otransformer_block_13/multi_head_attention_13/query/einsum/Einsum/ReadVariableOp?Etransformer_block_13/multi_head_attention_13/value/add/ReadVariableOp?Otransformer_block_13/multi_head_attention_13/value/einsum/Einsum/ReadVariableOp?Btransformer_block_13/sequential_13/dense_52/BiasAdd/ReadVariableOp?Dtransformer_block_13/sequential_13/dense_52/Tensordot/ReadVariableOp?Btransformer_block_13/sequential_13/dense_53/BiasAdd/ReadVariableOp?Dtransformer_block_13/sequential_13/dense_53/Tensordot/ReadVariableOp[
%token_and_position_embedding_13/ShapeShapeinputs*
T0*
_output_shapes
:?
3token_and_position_embedding_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????
5token_and_position_embedding_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
5token_and_position_embedding_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
-token_and_position_embedding_13/strided_sliceStridedSlice.token_and_position_embedding_13/Shape:output:0<token_and_position_embedding_13/strided_slice/stack:output:0>token_and_position_embedding_13/strided_slice/stack_1:output:0>token_and_position_embedding_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
+token_and_position_embedding_13/range/startConst*
_output_shapes
: *
dtype0*
value	B : m
+token_and_position_embedding_13/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
%token_and_position_embedding_13/rangeRange4token_and_position_embedding_13/range/start:output:06token_and_position_embedding_13/strided_slice:output:04token_and_position_embedding_13/range/delta:output:0*
_output_shapes	
:??
=token_and_position_embedding_13/embedding_27/embedding_lookupResourceGatherCtoken_and_position_embedding_13_embedding_27_embedding_lookup_82550.token_and_position_embedding_13/range:output:0*
Tindices0*V
_classL
JHloc:@token_and_position_embedding_13/embedding_27/embedding_lookup/82550* 
_output_shapes
:
??*
dtype0?
Ftoken_and_position_embedding_13/embedding_27/embedding_lookup/IdentityIdentityFtoken_and_position_embedding_13/embedding_27/embedding_lookup:output:0*
T0*V
_classL
JHloc:@token_and_position_embedding_13/embedding_27/embedding_lookup/82550* 
_output_shapes
:
???
Htoken_and_position_embedding_13/embedding_27/embedding_lookup/Identity_1IdentityOtoken_and_position_embedding_13/embedding_27/embedding_lookup/Identity:output:0*
T0* 
_output_shapes
:
???
1token_and_position_embedding_13/embedding_26/CastCastinputs*

DstT0*

SrcT0*(
_output_shapes
:???????????
=token_and_position_embedding_13/embedding_26/embedding_lookupResourceGatherCtoken_and_position_embedding_13_embedding_26_embedding_lookup_825565token_and_position_embedding_13/embedding_26/Cast:y:0*
Tindices0*V
_classL
JHloc:@token_and_position_embedding_13/embedding_26/embedding_lookup/82556*-
_output_shapes
:???????????*
dtype0?
Ftoken_and_position_embedding_13/embedding_26/embedding_lookup/IdentityIdentityFtoken_and_position_embedding_13/embedding_26/embedding_lookup:output:0*
T0*V
_classL
JHloc:@token_and_position_embedding_13/embedding_26/embedding_lookup/82556*-
_output_shapes
:????????????
Htoken_and_position_embedding_13/embedding_26/embedding_lookup/Identity_1IdentityOtoken_and_position_embedding_13/embedding_26/embedding_lookup/Identity:output:0*
T0*-
_output_shapes
:????????????
#token_and_position_embedding_13/addAddV2Qtoken_and_position_embedding_13/embedding_26/embedding_lookup/Identity_1:output:0Qtoken_and_position_embedding_13/embedding_27/embedding_lookup/Identity_1:output:0*
T0*-
_output_shapes
:????????????
Otransformer_block_13/multi_head_attention_13/query/einsum/Einsum/ReadVariableOpReadVariableOpXtransformer_block_13_multi_head_attention_13_query_einsum_einsum_readvariableop_resource*$
_output_shapes
:??*
dtype0?
@transformer_block_13/multi_head_attention_13/query/einsum/EinsumEinsum'token_and_position_embedding_13/add:z:0Wtransformer_block_13/multi_head_attention_13/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*1
_output_shapes
:???????????*
equationabc,cde->abde?
Etransformer_block_13/multi_head_attention_13/query/add/ReadVariableOpReadVariableOpNtransformer_block_13_multi_head_attention_13_query_add_readvariableop_resource*
_output_shapes
:	?*
dtype0?
6transformer_block_13/multi_head_attention_13/query/addAddV2Itransformer_block_13/multi_head_attention_13/query/einsum/Einsum:output:0Mtransformer_block_13/multi_head_attention_13/query/add/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
Mtransformer_block_13/multi_head_attention_13/key/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_13_multi_head_attention_13_key_einsum_einsum_readvariableop_resource*$
_output_shapes
:??*
dtype0?
>transformer_block_13/multi_head_attention_13/key/einsum/EinsumEinsum'token_and_position_embedding_13/add:z:0Utransformer_block_13/multi_head_attention_13/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*1
_output_shapes
:???????????*
equationabc,cde->abde?
Ctransformer_block_13/multi_head_attention_13/key/add/ReadVariableOpReadVariableOpLtransformer_block_13_multi_head_attention_13_key_add_readvariableop_resource*
_output_shapes
:	?*
dtype0?
4transformer_block_13/multi_head_attention_13/key/addAddV2Gtransformer_block_13/multi_head_attention_13/key/einsum/Einsum:output:0Ktransformer_block_13/multi_head_attention_13/key/add/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
Otransformer_block_13/multi_head_attention_13/value/einsum/Einsum/ReadVariableOpReadVariableOpXtransformer_block_13_multi_head_attention_13_value_einsum_einsum_readvariableop_resource*$
_output_shapes
:??*
dtype0?
@transformer_block_13/multi_head_attention_13/value/einsum/EinsumEinsum'token_and_position_embedding_13/add:z:0Wtransformer_block_13/multi_head_attention_13/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*1
_output_shapes
:???????????*
equationabc,cde->abde?
Etransformer_block_13/multi_head_attention_13/value/add/ReadVariableOpReadVariableOpNtransformer_block_13_multi_head_attention_13_value_add_readvariableop_resource*
_output_shapes
:	?*
dtype0?
6transformer_block_13/multi_head_attention_13/value/addAddV2Itransformer_block_13/multi_head_attention_13/value/einsum/Einsum:output:0Mtransformer_block_13/multi_head_attention_13/value/add/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????w
2transformer_block_13/multi_head_attention_13/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *?А=?
0transformer_block_13/multi_head_attention_13/MulMul:transformer_block_13/multi_head_attention_13/query/add:z:0;transformer_block_13/multi_head_attention_13/Mul/y:output:0*
T0*1
_output_shapes
:????????????
:transformer_block_13/multi_head_attention_13/einsum/EinsumEinsum8transformer_block_13/multi_head_attention_13/key/add:z:04transformer_block_13/multi_head_attention_13/Mul:z:0*
N*
T0*1
_output_shapes
:???????????*
equationaecd,abcd->acbe?
<transformer_block_13/multi_head_attention_13/softmax/SoftmaxSoftmaxCtransformer_block_13/multi_head_attention_13/einsum/Einsum:output:0*
T0*1
_output_shapes
:????????????
=transformer_block_13/multi_head_attention_13/dropout/IdentityIdentityFtransformer_block_13/multi_head_attention_13/softmax/Softmax:softmax:0*
T0*1
_output_shapes
:????????????
<transformer_block_13/multi_head_attention_13/einsum_1/EinsumEinsumFtransformer_block_13/multi_head_attention_13/dropout/Identity:output:0:transformer_block_13/multi_head_attention_13/value/add:z:0*
N*
T0*1
_output_shapes
:???????????*
equationacbe,aecd->abcd?
Ztransformer_block_13/multi_head_attention_13/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpctransformer_block_13_multi_head_attention_13_attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:??*
dtype0?
Ktransformer_block_13/multi_head_attention_13/attention_output/einsum/EinsumEinsumEtransformer_block_13/multi_head_attention_13/einsum_1/Einsum:output:0btransformer_block_13/multi_head_attention_13/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*-
_output_shapes
:???????????*
equationabcd,cde->abe?
Ptransformer_block_13/multi_head_attention_13/attention_output/add/ReadVariableOpReadVariableOpYtransformer_block_13_multi_head_attention_13_attention_output_add_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Atransformer_block_13/multi_head_attention_13/attention_output/addAddV2Ttransformer_block_13/multi_head_attention_13/attention_output/einsum/Einsum:output:0Xtransformer_block_13/multi_head_attention_13/attention_output/add/ReadVariableOp:value:0*
T0*-
_output_shapes
:????????????
(transformer_block_13/dropout_55/IdentityIdentityEtransformer_block_13/multi_head_attention_13/attention_output/add:z:0*
T0*-
_output_shapes
:????????????
transformer_block_13/addAddV2'token_and_position_embedding_13/add:z:01transformer_block_13/dropout_55/Identity:output:0*
T0*-
_output_shapes
:???????????}
1transformer_block_13/layer_normalization_26/ShapeShapetransformer_block_13/add:z:0*
T0*
_output_shapes
:?
?transformer_block_13/layer_normalization_26/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Atransformer_block_13/layer_normalization_26/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Atransformer_block_13/layer_normalization_26/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
9transformer_block_13/layer_normalization_26/strided_sliceStridedSlice:transformer_block_13/layer_normalization_26/Shape:output:0Htransformer_block_13/layer_normalization_26/strided_slice/stack:output:0Jtransformer_block_13/layer_normalization_26/strided_slice/stack_1:output:0Jtransformer_block_13/layer_normalization_26/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
1transformer_block_13/layer_normalization_26/mul/xConst*
_output_shapes
: *
dtype0*
value	B :?
/transformer_block_13/layer_normalization_26/mulMul:transformer_block_13/layer_normalization_26/mul/x:output:0Btransformer_block_13/layer_normalization_26/strided_slice:output:0*
T0*
_output_shapes
: ?
Atransformer_block_13/layer_normalization_26/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Ctransformer_block_13/layer_normalization_26/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Ctransformer_block_13/layer_normalization_26/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;transformer_block_13/layer_normalization_26/strided_slice_1StridedSlice:transformer_block_13/layer_normalization_26/Shape:output:0Jtransformer_block_13/layer_normalization_26/strided_slice_1/stack:output:0Ltransformer_block_13/layer_normalization_26/strided_slice_1/stack_1:output:0Ltransformer_block_13/layer_normalization_26/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
1transformer_block_13/layer_normalization_26/mul_1Mul3transformer_block_13/layer_normalization_26/mul:z:0Dtransformer_block_13/layer_normalization_26/strided_slice_1:output:0*
T0*
_output_shapes
: ?
Atransformer_block_13/layer_normalization_26/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Ctransformer_block_13/layer_normalization_26/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Ctransformer_block_13/layer_normalization_26/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;transformer_block_13/layer_normalization_26/strided_slice_2StridedSlice:transformer_block_13/layer_normalization_26/Shape:output:0Jtransformer_block_13/layer_normalization_26/strided_slice_2/stack:output:0Ltransformer_block_13/layer_normalization_26/strided_slice_2/stack_1:output:0Ltransformer_block_13/layer_normalization_26/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3transformer_block_13/layer_normalization_26/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :?
1transformer_block_13/layer_normalization_26/mul_2Mul<transformer_block_13/layer_normalization_26/mul_2/x:output:0Dtransformer_block_13/layer_normalization_26/strided_slice_2:output:0*
T0*
_output_shapes
: }
;transformer_block_13/layer_normalization_26/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :}
;transformer_block_13/layer_normalization_26/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
9transformer_block_13/layer_normalization_26/Reshape/shapePackDtransformer_block_13/layer_normalization_26/Reshape/shape/0:output:05transformer_block_13/layer_normalization_26/mul_1:z:05transformer_block_13/layer_normalization_26/mul_2:z:0Dtransformer_block_13/layer_normalization_26/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
3transformer_block_13/layer_normalization_26/ReshapeReshapetransformer_block_13/add:z:0Btransformer_block_13/layer_normalization_26/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????z
7transformer_block_13/layer_normalization_26/ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :??
5transformer_block_13/layer_normalization_26/ones/LessLess5transformer_block_13/layer_normalization_26/mul_1:z:0@transformer_block_13/layer_normalization_26/ones/Less/y:output:0*
T0*
_output_shapes
: ?
7transformer_block_13/layer_normalization_26/ones/packedPack5transformer_block_13/layer_normalization_26/mul_1:z:0*
N*
T0*
_output_shapes
:{
6transformer_block_13/layer_normalization_26/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
0transformer_block_13/layer_normalization_26/onesFill@transformer_block_13/layer_normalization_26/ones/packed:output:0?transformer_block_13/layer_normalization_26/ones/Const:output:0*
T0*#
_output_shapes
:?????????{
8transformer_block_13/layer_normalization_26/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :??
6transformer_block_13/layer_normalization_26/zeros/LessLess5transformer_block_13/layer_normalization_26/mul_1:z:0Atransformer_block_13/layer_normalization_26/zeros/Less/y:output:0*
T0*
_output_shapes
: ?
8transformer_block_13/layer_normalization_26/zeros/packedPack5transformer_block_13/layer_normalization_26/mul_1:z:0*
N*
T0*
_output_shapes
:|
7transformer_block_13/layer_normalization_26/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
1transformer_block_13/layer_normalization_26/zerosFillAtransformer_block_13/layer_normalization_26/zeros/packed:output:0@transformer_block_13/layer_normalization_26/zeros/Const:output:0*
T0*#
_output_shapes
:?????????t
1transformer_block_13/layer_normalization_26/ConstConst*
_output_shapes
: *
dtype0*
valueB v
3transformer_block_13/layer_normalization_26/Const_1Const*
_output_shapes
: *
dtype0*
valueB ?
<transformer_block_13/layer_normalization_26/FusedBatchNormV3FusedBatchNormV3<transformer_block_13/layer_normalization_26/Reshape:output:09transformer_block_13/layer_normalization_26/ones:output:0:transformer_block_13/layer_normalization_26/zeros:output:0:transformer_block_13/layer_normalization_26/Const:output:0<transformer_block_13/layer_normalization_26/Const_1:output:0*
T0*
U0*p
_output_shapes^
\:??????????:?????????:?????????:?????????:?????????:*
data_formatNCHW?
5transformer_block_13/layer_normalization_26/Reshape_1Reshape@transformer_block_13/layer_normalization_26/FusedBatchNormV3:y:0:transformer_block_13/layer_normalization_26/Shape:output:0*
T0*-
_output_shapes
:????????????
@transformer_block_13/layer_normalization_26/mul_3/ReadVariableOpReadVariableOpItransformer_block_13_layer_normalization_26_mul_3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
1transformer_block_13/layer_normalization_26/mul_3Mul>transformer_block_13/layer_normalization_26/Reshape_1:output:0Htransformer_block_13/layer_normalization_26/mul_3/ReadVariableOp:value:0*
T0*-
_output_shapes
:????????????
>transformer_block_13/layer_normalization_26/add/ReadVariableOpReadVariableOpGtransformer_block_13_layer_normalization_26_add_readvariableop_resource*
_output_shapes	
:?*
dtype0?
/transformer_block_13/layer_normalization_26/addAddV25transformer_block_13/layer_normalization_26/mul_3:z:0Ftransformer_block_13/layer_normalization_26/add/ReadVariableOp:value:0*
T0*-
_output_shapes
:????????????
Dtransformer_block_13/sequential_13/dense_52/Tensordot/ReadVariableOpReadVariableOpMtransformer_block_13_sequential_13_dense_52_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
:transformer_block_13/sequential_13/dense_52/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
:transformer_block_13/sequential_13/dense_52/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
;transformer_block_13/sequential_13/dense_52/Tensordot/ShapeShape3transformer_block_13/layer_normalization_26/add:z:0*
T0*
_output_shapes
:?
Ctransformer_block_13/sequential_13/dense_52/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
>transformer_block_13/sequential_13/dense_52/Tensordot/GatherV2GatherV2Dtransformer_block_13/sequential_13/dense_52/Tensordot/Shape:output:0Ctransformer_block_13/sequential_13/dense_52/Tensordot/free:output:0Ltransformer_block_13/sequential_13/dense_52/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Etransformer_block_13/sequential_13/dense_52/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
@transformer_block_13/sequential_13/dense_52/Tensordot/GatherV2_1GatherV2Dtransformer_block_13/sequential_13/dense_52/Tensordot/Shape:output:0Ctransformer_block_13/sequential_13/dense_52/Tensordot/axes:output:0Ntransformer_block_13/sequential_13/dense_52/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
;transformer_block_13/sequential_13/dense_52/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
:transformer_block_13/sequential_13/dense_52/Tensordot/ProdProdGtransformer_block_13/sequential_13/dense_52/Tensordot/GatherV2:output:0Dtransformer_block_13/sequential_13/dense_52/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
=transformer_block_13/sequential_13/dense_52/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
<transformer_block_13/sequential_13/dense_52/Tensordot/Prod_1ProdItransformer_block_13/sequential_13/dense_52/Tensordot/GatherV2_1:output:0Ftransformer_block_13/sequential_13/dense_52/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ?
Atransformer_block_13/sequential_13/dense_52/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
<transformer_block_13/sequential_13/dense_52/Tensordot/concatConcatV2Ctransformer_block_13/sequential_13/dense_52/Tensordot/free:output:0Ctransformer_block_13/sequential_13/dense_52/Tensordot/axes:output:0Jtransformer_block_13/sequential_13/dense_52/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
;transformer_block_13/sequential_13/dense_52/Tensordot/stackPackCtransformer_block_13/sequential_13/dense_52/Tensordot/Prod:output:0Etransformer_block_13/sequential_13/dense_52/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
?transformer_block_13/sequential_13/dense_52/Tensordot/transpose	Transpose3transformer_block_13/layer_normalization_26/add:z:0Etransformer_block_13/sequential_13/dense_52/Tensordot/concat:output:0*
T0*-
_output_shapes
:????????????
=transformer_block_13/sequential_13/dense_52/Tensordot/ReshapeReshapeCtransformer_block_13/sequential_13/dense_52/Tensordot/transpose:y:0Dtransformer_block_13/sequential_13/dense_52/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
<transformer_block_13/sequential_13/dense_52/Tensordot/MatMulMatMulFtransformer_block_13/sequential_13/dense_52/Tensordot/Reshape:output:0Ltransformer_block_13/sequential_13/dense_52/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
=transformer_block_13/sequential_13/dense_52/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:??
Ctransformer_block_13/sequential_13/dense_52/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
>transformer_block_13/sequential_13/dense_52/Tensordot/concat_1ConcatV2Gtransformer_block_13/sequential_13/dense_52/Tensordot/GatherV2:output:0Ftransformer_block_13/sequential_13/dense_52/Tensordot/Const_2:output:0Ltransformer_block_13/sequential_13/dense_52/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
5transformer_block_13/sequential_13/dense_52/TensordotReshapeFtransformer_block_13/sequential_13/dense_52/Tensordot/MatMul:product:0Gtransformer_block_13/sequential_13/dense_52/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:????????????
Btransformer_block_13/sequential_13/dense_52/BiasAdd/ReadVariableOpReadVariableOpKtransformer_block_13_sequential_13_dense_52_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
3transformer_block_13/sequential_13/dense_52/BiasAddBiasAdd>transformer_block_13/sequential_13/dense_52/Tensordot:output:0Jtransformer_block_13/sequential_13/dense_52/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:????????????
0transformer_block_13/sequential_13/dense_52/ReluRelu<transformer_block_13/sequential_13/dense_52/BiasAdd:output:0*
T0*-
_output_shapes
:????????????
Dtransformer_block_13/sequential_13/dense_53/Tensordot/ReadVariableOpReadVariableOpMtransformer_block_13_sequential_13_dense_53_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
:transformer_block_13/sequential_13/dense_53/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
:transformer_block_13/sequential_13/dense_53/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
;transformer_block_13/sequential_13/dense_53/Tensordot/ShapeShape>transformer_block_13/sequential_13/dense_52/Relu:activations:0*
T0*
_output_shapes
:?
Ctransformer_block_13/sequential_13/dense_53/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
>transformer_block_13/sequential_13/dense_53/Tensordot/GatherV2GatherV2Dtransformer_block_13/sequential_13/dense_53/Tensordot/Shape:output:0Ctransformer_block_13/sequential_13/dense_53/Tensordot/free:output:0Ltransformer_block_13/sequential_13/dense_53/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Etransformer_block_13/sequential_13/dense_53/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
@transformer_block_13/sequential_13/dense_53/Tensordot/GatherV2_1GatherV2Dtransformer_block_13/sequential_13/dense_53/Tensordot/Shape:output:0Ctransformer_block_13/sequential_13/dense_53/Tensordot/axes:output:0Ntransformer_block_13/sequential_13/dense_53/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
;transformer_block_13/sequential_13/dense_53/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
:transformer_block_13/sequential_13/dense_53/Tensordot/ProdProdGtransformer_block_13/sequential_13/dense_53/Tensordot/GatherV2:output:0Dtransformer_block_13/sequential_13/dense_53/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
=transformer_block_13/sequential_13/dense_53/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
<transformer_block_13/sequential_13/dense_53/Tensordot/Prod_1ProdItransformer_block_13/sequential_13/dense_53/Tensordot/GatherV2_1:output:0Ftransformer_block_13/sequential_13/dense_53/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ?
Atransformer_block_13/sequential_13/dense_53/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
<transformer_block_13/sequential_13/dense_53/Tensordot/concatConcatV2Ctransformer_block_13/sequential_13/dense_53/Tensordot/free:output:0Ctransformer_block_13/sequential_13/dense_53/Tensordot/axes:output:0Jtransformer_block_13/sequential_13/dense_53/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
;transformer_block_13/sequential_13/dense_53/Tensordot/stackPackCtransformer_block_13/sequential_13/dense_53/Tensordot/Prod:output:0Etransformer_block_13/sequential_13/dense_53/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
?transformer_block_13/sequential_13/dense_53/Tensordot/transpose	Transpose>transformer_block_13/sequential_13/dense_52/Relu:activations:0Etransformer_block_13/sequential_13/dense_53/Tensordot/concat:output:0*
T0*-
_output_shapes
:????????????
=transformer_block_13/sequential_13/dense_53/Tensordot/ReshapeReshapeCtransformer_block_13/sequential_13/dense_53/Tensordot/transpose:y:0Dtransformer_block_13/sequential_13/dense_53/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
<transformer_block_13/sequential_13/dense_53/Tensordot/MatMulMatMulFtransformer_block_13/sequential_13/dense_53/Tensordot/Reshape:output:0Ltransformer_block_13/sequential_13/dense_53/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
=transformer_block_13/sequential_13/dense_53/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:??
Ctransformer_block_13/sequential_13/dense_53/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
>transformer_block_13/sequential_13/dense_53/Tensordot/concat_1ConcatV2Gtransformer_block_13/sequential_13/dense_53/Tensordot/GatherV2:output:0Ftransformer_block_13/sequential_13/dense_53/Tensordot/Const_2:output:0Ltransformer_block_13/sequential_13/dense_53/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
5transformer_block_13/sequential_13/dense_53/TensordotReshapeFtransformer_block_13/sequential_13/dense_53/Tensordot/MatMul:product:0Gtransformer_block_13/sequential_13/dense_53/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:????????????
Btransformer_block_13/sequential_13/dense_53/BiasAdd/ReadVariableOpReadVariableOpKtransformer_block_13_sequential_13_dense_53_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
3transformer_block_13/sequential_13/dense_53/BiasAddBiasAdd>transformer_block_13/sequential_13/dense_53/Tensordot:output:0Jtransformer_block_13/sequential_13/dense_53/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:????????????
(transformer_block_13/dropout_56/IdentityIdentity<transformer_block_13/sequential_13/dense_53/BiasAdd:output:0*
T0*-
_output_shapes
:????????????
transformer_block_13/add_1AddV23transformer_block_13/layer_normalization_26/add:z:01transformer_block_13/dropout_56/Identity:output:0*
T0*-
_output_shapes
:???????????
1transformer_block_13/layer_normalization_27/ShapeShapetransformer_block_13/add_1:z:0*
T0*
_output_shapes
:?
?transformer_block_13/layer_normalization_27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Atransformer_block_13/layer_normalization_27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Atransformer_block_13/layer_normalization_27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
9transformer_block_13/layer_normalization_27/strided_sliceStridedSlice:transformer_block_13/layer_normalization_27/Shape:output:0Htransformer_block_13/layer_normalization_27/strided_slice/stack:output:0Jtransformer_block_13/layer_normalization_27/strided_slice/stack_1:output:0Jtransformer_block_13/layer_normalization_27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
1transformer_block_13/layer_normalization_27/mul/xConst*
_output_shapes
: *
dtype0*
value	B :?
/transformer_block_13/layer_normalization_27/mulMul:transformer_block_13/layer_normalization_27/mul/x:output:0Btransformer_block_13/layer_normalization_27/strided_slice:output:0*
T0*
_output_shapes
: ?
Atransformer_block_13/layer_normalization_27/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Ctransformer_block_13/layer_normalization_27/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Ctransformer_block_13/layer_normalization_27/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;transformer_block_13/layer_normalization_27/strided_slice_1StridedSlice:transformer_block_13/layer_normalization_27/Shape:output:0Jtransformer_block_13/layer_normalization_27/strided_slice_1/stack:output:0Ltransformer_block_13/layer_normalization_27/strided_slice_1/stack_1:output:0Ltransformer_block_13/layer_normalization_27/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
1transformer_block_13/layer_normalization_27/mul_1Mul3transformer_block_13/layer_normalization_27/mul:z:0Dtransformer_block_13/layer_normalization_27/strided_slice_1:output:0*
T0*
_output_shapes
: ?
Atransformer_block_13/layer_normalization_27/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Ctransformer_block_13/layer_normalization_27/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Ctransformer_block_13/layer_normalization_27/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;transformer_block_13/layer_normalization_27/strided_slice_2StridedSlice:transformer_block_13/layer_normalization_27/Shape:output:0Jtransformer_block_13/layer_normalization_27/strided_slice_2/stack:output:0Ltransformer_block_13/layer_normalization_27/strided_slice_2/stack_1:output:0Ltransformer_block_13/layer_normalization_27/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3transformer_block_13/layer_normalization_27/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :?
1transformer_block_13/layer_normalization_27/mul_2Mul<transformer_block_13/layer_normalization_27/mul_2/x:output:0Dtransformer_block_13/layer_normalization_27/strided_slice_2:output:0*
T0*
_output_shapes
: }
;transformer_block_13/layer_normalization_27/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :}
;transformer_block_13/layer_normalization_27/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
9transformer_block_13/layer_normalization_27/Reshape/shapePackDtransformer_block_13/layer_normalization_27/Reshape/shape/0:output:05transformer_block_13/layer_normalization_27/mul_1:z:05transformer_block_13/layer_normalization_27/mul_2:z:0Dtransformer_block_13/layer_normalization_27/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
3transformer_block_13/layer_normalization_27/ReshapeReshapetransformer_block_13/add_1:z:0Btransformer_block_13/layer_normalization_27/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????z
7transformer_block_13/layer_normalization_27/ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :??
5transformer_block_13/layer_normalization_27/ones/LessLess5transformer_block_13/layer_normalization_27/mul_1:z:0@transformer_block_13/layer_normalization_27/ones/Less/y:output:0*
T0*
_output_shapes
: ?
7transformer_block_13/layer_normalization_27/ones/packedPack5transformer_block_13/layer_normalization_27/mul_1:z:0*
N*
T0*
_output_shapes
:{
6transformer_block_13/layer_normalization_27/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
0transformer_block_13/layer_normalization_27/onesFill@transformer_block_13/layer_normalization_27/ones/packed:output:0?transformer_block_13/layer_normalization_27/ones/Const:output:0*
T0*#
_output_shapes
:?????????{
8transformer_block_13/layer_normalization_27/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :??
6transformer_block_13/layer_normalization_27/zeros/LessLess5transformer_block_13/layer_normalization_27/mul_1:z:0Atransformer_block_13/layer_normalization_27/zeros/Less/y:output:0*
T0*
_output_shapes
: ?
8transformer_block_13/layer_normalization_27/zeros/packedPack5transformer_block_13/layer_normalization_27/mul_1:z:0*
N*
T0*
_output_shapes
:|
7transformer_block_13/layer_normalization_27/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
1transformer_block_13/layer_normalization_27/zerosFillAtransformer_block_13/layer_normalization_27/zeros/packed:output:0@transformer_block_13/layer_normalization_27/zeros/Const:output:0*
T0*#
_output_shapes
:?????????t
1transformer_block_13/layer_normalization_27/ConstConst*
_output_shapes
: *
dtype0*
valueB v
3transformer_block_13/layer_normalization_27/Const_1Const*
_output_shapes
: *
dtype0*
valueB ?
<transformer_block_13/layer_normalization_27/FusedBatchNormV3FusedBatchNormV3<transformer_block_13/layer_normalization_27/Reshape:output:09transformer_block_13/layer_normalization_27/ones:output:0:transformer_block_13/layer_normalization_27/zeros:output:0:transformer_block_13/layer_normalization_27/Const:output:0<transformer_block_13/layer_normalization_27/Const_1:output:0*
T0*
U0*p
_output_shapes^
\:??????????:?????????:?????????:?????????:?????????:*
data_formatNCHW?
5transformer_block_13/layer_normalization_27/Reshape_1Reshape@transformer_block_13/layer_normalization_27/FusedBatchNormV3:y:0:transformer_block_13/layer_normalization_27/Shape:output:0*
T0*-
_output_shapes
:????????????
@transformer_block_13/layer_normalization_27/mul_3/ReadVariableOpReadVariableOpItransformer_block_13_layer_normalization_27_mul_3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
1transformer_block_13/layer_normalization_27/mul_3Mul>transformer_block_13/layer_normalization_27/Reshape_1:output:0Htransformer_block_13/layer_normalization_27/mul_3/ReadVariableOp:value:0*
T0*-
_output_shapes
:????????????
>transformer_block_13/layer_normalization_27/add/ReadVariableOpReadVariableOpGtransformer_block_13_layer_normalization_27_add_readvariableop_resource*
_output_shapes	
:?*
dtype0?
/transformer_block_13/layer_normalization_27/addAddV25transformer_block_13/layer_normalization_27/mul_3:z:0Ftransformer_block_13/layer_normalization_27/add/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????t
2global_average_pooling1d_13/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
 global_average_pooling1d_13/MeanMean3transformer_block_13/layer_normalization_27/add:z:0;global_average_pooling1d_13/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:??????????}
dropout_57/IdentityIdentity)global_average_pooling1d_13/Mean:output:0*
T0*(
_output_shapes
:???????????
dense_54/MatMul/ReadVariableOpReadVariableOp'dense_54_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense_54/MatMulMatMuldropout_57/Identity:output:0&dense_54/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_54/BiasAdd/ReadVariableOpReadVariableOp(dense_54_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_54/BiasAddBiasAdddense_54/MatMul:product:0'dense_54/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????b
dense_54/ReluReludense_54/BiasAdd:output:0*
T0*'
_output_shapes
:?????????n
dropout_58/IdentityIdentitydense_54/Relu:activations:0*
T0*'
_output_shapes
:??????????
dense_55/MatMul/ReadVariableOpReadVariableOp'dense_55_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_55/MatMulMatMuldropout_58/Identity:output:0&dense_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_55/BiasAdd/ReadVariableOpReadVariableOp(dense_55_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_55/BiasAddBiasAdddense_55/MatMul:product:0'dense_55/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
dense_55/SoftmaxSoftmaxdense_55/BiasAdd:output:0*
T0*'
_output_shapes
:?????????i
IdentityIdentitydense_55/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_54/BiasAdd/ReadVariableOp^dense_54/MatMul/ReadVariableOp ^dense_55/BiasAdd/ReadVariableOp^dense_55/MatMul/ReadVariableOp>^token_and_position_embedding_13/embedding_26/embedding_lookup>^token_and_position_embedding_13/embedding_27/embedding_lookup?^transformer_block_13/layer_normalization_26/add/ReadVariableOpA^transformer_block_13/layer_normalization_26/mul_3/ReadVariableOp?^transformer_block_13/layer_normalization_27/add/ReadVariableOpA^transformer_block_13/layer_normalization_27/mul_3/ReadVariableOpQ^transformer_block_13/multi_head_attention_13/attention_output/add/ReadVariableOp[^transformer_block_13/multi_head_attention_13/attention_output/einsum/Einsum/ReadVariableOpD^transformer_block_13/multi_head_attention_13/key/add/ReadVariableOpN^transformer_block_13/multi_head_attention_13/key/einsum/Einsum/ReadVariableOpF^transformer_block_13/multi_head_attention_13/query/add/ReadVariableOpP^transformer_block_13/multi_head_attention_13/query/einsum/Einsum/ReadVariableOpF^transformer_block_13/multi_head_attention_13/value/add/ReadVariableOpP^transformer_block_13/multi_head_attention_13/value/einsum/Einsum/ReadVariableOpC^transformer_block_13/sequential_13/dense_52/BiasAdd/ReadVariableOpE^transformer_block_13/sequential_13/dense_52/Tensordot/ReadVariableOpC^transformer_block_13/sequential_13/dense_53/BiasAdd/ReadVariableOpE^transformer_block_13/sequential_13/dense_53/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:??????????: : : : : : : : : : : : : : : : : : : : : : 2B
dense_54/BiasAdd/ReadVariableOpdense_54/BiasAdd/ReadVariableOp2@
dense_54/MatMul/ReadVariableOpdense_54/MatMul/ReadVariableOp2B
dense_55/BiasAdd/ReadVariableOpdense_55/BiasAdd/ReadVariableOp2@
dense_55/MatMul/ReadVariableOpdense_55/MatMul/ReadVariableOp2~
=token_and_position_embedding_13/embedding_26/embedding_lookup=token_and_position_embedding_13/embedding_26/embedding_lookup2~
=token_and_position_embedding_13/embedding_27/embedding_lookup=token_and_position_embedding_13/embedding_27/embedding_lookup2?
>transformer_block_13/layer_normalization_26/add/ReadVariableOp>transformer_block_13/layer_normalization_26/add/ReadVariableOp2?
@transformer_block_13/layer_normalization_26/mul_3/ReadVariableOp@transformer_block_13/layer_normalization_26/mul_3/ReadVariableOp2?
>transformer_block_13/layer_normalization_27/add/ReadVariableOp>transformer_block_13/layer_normalization_27/add/ReadVariableOp2?
@transformer_block_13/layer_normalization_27/mul_3/ReadVariableOp@transformer_block_13/layer_normalization_27/mul_3/ReadVariableOp2?
Ptransformer_block_13/multi_head_attention_13/attention_output/add/ReadVariableOpPtransformer_block_13/multi_head_attention_13/attention_output/add/ReadVariableOp2?
Ztransformer_block_13/multi_head_attention_13/attention_output/einsum/Einsum/ReadVariableOpZtransformer_block_13/multi_head_attention_13/attention_output/einsum/Einsum/ReadVariableOp2?
Ctransformer_block_13/multi_head_attention_13/key/add/ReadVariableOpCtransformer_block_13/multi_head_attention_13/key/add/ReadVariableOp2?
Mtransformer_block_13/multi_head_attention_13/key/einsum/Einsum/ReadVariableOpMtransformer_block_13/multi_head_attention_13/key/einsum/Einsum/ReadVariableOp2?
Etransformer_block_13/multi_head_attention_13/query/add/ReadVariableOpEtransformer_block_13/multi_head_attention_13/query/add/ReadVariableOp2?
Otransformer_block_13/multi_head_attention_13/query/einsum/Einsum/ReadVariableOpOtransformer_block_13/multi_head_attention_13/query/einsum/Einsum/ReadVariableOp2?
Etransformer_block_13/multi_head_attention_13/value/add/ReadVariableOpEtransformer_block_13/multi_head_attention_13/value/add/ReadVariableOp2?
Otransformer_block_13/multi_head_attention_13/value/einsum/Einsum/ReadVariableOpOtransformer_block_13/multi_head_attention_13/value/einsum/Einsum/ReadVariableOp2?
Btransformer_block_13/sequential_13/dense_52/BiasAdd/ReadVariableOpBtransformer_block_13/sequential_13/dense_52/BiasAdd/ReadVariableOp2?
Dtransformer_block_13/sequential_13/dense_52/Tensordot/ReadVariableOpDtransformer_block_13/sequential_13/dense_52/Tensordot/ReadVariableOp2?
Btransformer_block_13/sequential_13/dense_53/BiasAdd/ReadVariableOpBtransformer_block_13/sequential_13/dense_53/BiasAdd/ReadVariableOp2?
Dtransformer_block_13/sequential_13/dense_53/Tensordot/ReadVariableOpDtransformer_block_13/sequential_13/dense_53/Tensordot/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
Z__inference_token_and_position_embedding_13_layer_call_and_return_conditional_losses_83045
x7
#embedding_27_embedding_lookup_83032:
??7
#embedding_26_embedding_lookup_83038:
?)?
identity??embedding_26/embedding_lookup?embedding_27/embedding_lookup6
ShapeShapex*
T0*
_output_shapes
:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :o
rangeRangerange/start:output:0strided_slice:output:0range/delta:output:0*
_output_shapes	
:??
embedding_27/embedding_lookupResourceGather#embedding_27_embedding_lookup_83032range:output:0*
Tindices0*6
_class,
*(loc:@embedding_27/embedding_lookup/83032* 
_output_shapes
:
??*
dtype0?
&embedding_27/embedding_lookup/IdentityIdentity&embedding_27/embedding_lookup:output:0*
T0*6
_class,
*(loc:@embedding_27/embedding_lookup/83032* 
_output_shapes
:
???
(embedding_27/embedding_lookup/Identity_1Identity/embedding_27/embedding_lookup/Identity:output:0*
T0* 
_output_shapes
:
??^
embedding_26/CastCastx*

DstT0*

SrcT0*(
_output_shapes
:???????????
embedding_26/embedding_lookupResourceGather#embedding_26_embedding_lookup_83038embedding_26/Cast:y:0*
Tindices0*6
_class,
*(loc:@embedding_26/embedding_lookup/83038*-
_output_shapes
:???????????*
dtype0?
&embedding_26/embedding_lookup/IdentityIdentity&embedding_26/embedding_lookup:output:0*
T0*6
_class,
*(loc:@embedding_26/embedding_lookup/83038*-
_output_shapes
:????????????
(embedding_26/embedding_lookup/Identity_1Identity/embedding_26/embedding_lookup/Identity:output:0*
T0*-
_output_shapes
:????????????
addAddV21embedding_26/embedding_lookup/Identity_1:output:01embedding_27/embedding_lookup/Identity_1:output:0*
T0*-
_output_shapes
:???????????\
IdentityIdentityadd:z:0^NoOp*
T0*-
_output_shapes
:????????????
NoOpNoOp^embedding_26/embedding_lookup^embedding_27/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2>
embedding_26/embedding_lookupembedding_26/embedding_lookup2>
embedding_27/embedding_lookupembedding_27/embedding_lookup:K G
(
_output_shapes
:??????????

_user_specified_namex
?
?
C__inference_dense_52_layer_call_and_return_conditional_losses_81191

inputs5
!tensordot_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:{
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*-
_output_shapes
:????????????
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*-
_output_shapes
:???????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????V
ReluReluBiasAdd:output:0*
T0*-
_output_shapes
:???????????g
IdentityIdentityRelu:activations:0^NoOp*
T0*-
_output_shapes
:???????????z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?
C__inference_model_13_layer_call_and_return_conditional_losses_83012

inputsW
Ctoken_and_position_embedding_13_embedding_27_embedding_lookup_82773:
??W
Ctoken_and_position_embedding_13_embedding_26_embedding_lookup_82779:
?)?p
Xtransformer_block_13_multi_head_attention_13_query_einsum_einsum_readvariableop_resource:??a
Ntransformer_block_13_multi_head_attention_13_query_add_readvariableop_resource:	?n
Vtransformer_block_13_multi_head_attention_13_key_einsum_einsum_readvariableop_resource:??_
Ltransformer_block_13_multi_head_attention_13_key_add_readvariableop_resource:	?p
Xtransformer_block_13_multi_head_attention_13_value_einsum_einsum_readvariableop_resource:??a
Ntransformer_block_13_multi_head_attention_13_value_add_readvariableop_resource:	?{
ctransformer_block_13_multi_head_attention_13_attention_output_einsum_einsum_readvariableop_resource:??h
Ytransformer_block_13_multi_head_attention_13_attention_output_add_readvariableop_resource:	?X
Itransformer_block_13_layer_normalization_26_mul_3_readvariableop_resource:	?V
Gtransformer_block_13_layer_normalization_26_add_readvariableop_resource:	?a
Mtransformer_block_13_sequential_13_dense_52_tensordot_readvariableop_resource:
??Z
Ktransformer_block_13_sequential_13_dense_52_biasadd_readvariableop_resource:	?a
Mtransformer_block_13_sequential_13_dense_53_tensordot_readvariableop_resource:
??Z
Ktransformer_block_13_sequential_13_dense_53_biasadd_readvariableop_resource:	?X
Itransformer_block_13_layer_normalization_27_mul_3_readvariableop_resource:	?V
Gtransformer_block_13_layer_normalization_27_add_readvariableop_resource:	?:
'dense_54_matmul_readvariableop_resource:	?6
(dense_54_biasadd_readvariableop_resource:9
'dense_55_matmul_readvariableop_resource:6
(dense_55_biasadd_readvariableop_resource:
identity??dense_54/BiasAdd/ReadVariableOp?dense_54/MatMul/ReadVariableOp?dense_55/BiasAdd/ReadVariableOp?dense_55/MatMul/ReadVariableOp?=token_and_position_embedding_13/embedding_26/embedding_lookup?=token_and_position_embedding_13/embedding_27/embedding_lookup?>transformer_block_13/layer_normalization_26/add/ReadVariableOp?@transformer_block_13/layer_normalization_26/mul_3/ReadVariableOp?>transformer_block_13/layer_normalization_27/add/ReadVariableOp?@transformer_block_13/layer_normalization_27/mul_3/ReadVariableOp?Ptransformer_block_13/multi_head_attention_13/attention_output/add/ReadVariableOp?Ztransformer_block_13/multi_head_attention_13/attention_output/einsum/Einsum/ReadVariableOp?Ctransformer_block_13/multi_head_attention_13/key/add/ReadVariableOp?Mtransformer_block_13/multi_head_attention_13/key/einsum/Einsum/ReadVariableOp?Etransformer_block_13/multi_head_attention_13/query/add/ReadVariableOp?Otransformer_block_13/multi_head_attention_13/query/einsum/Einsum/ReadVariableOp?Etransformer_block_13/multi_head_attention_13/value/add/ReadVariableOp?Otransformer_block_13/multi_head_attention_13/value/einsum/Einsum/ReadVariableOp?Btransformer_block_13/sequential_13/dense_52/BiasAdd/ReadVariableOp?Dtransformer_block_13/sequential_13/dense_52/Tensordot/ReadVariableOp?Btransformer_block_13/sequential_13/dense_53/BiasAdd/ReadVariableOp?Dtransformer_block_13/sequential_13/dense_53/Tensordot/ReadVariableOp[
%token_and_position_embedding_13/ShapeShapeinputs*
T0*
_output_shapes
:?
3token_and_position_embedding_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????
5token_and_position_embedding_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
5token_and_position_embedding_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
-token_and_position_embedding_13/strided_sliceStridedSlice.token_and_position_embedding_13/Shape:output:0<token_and_position_embedding_13/strided_slice/stack:output:0>token_and_position_embedding_13/strided_slice/stack_1:output:0>token_and_position_embedding_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
+token_and_position_embedding_13/range/startConst*
_output_shapes
: *
dtype0*
value	B : m
+token_and_position_embedding_13/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
%token_and_position_embedding_13/rangeRange4token_and_position_embedding_13/range/start:output:06token_and_position_embedding_13/strided_slice:output:04token_and_position_embedding_13/range/delta:output:0*
_output_shapes	
:??
=token_and_position_embedding_13/embedding_27/embedding_lookupResourceGatherCtoken_and_position_embedding_13_embedding_27_embedding_lookup_82773.token_and_position_embedding_13/range:output:0*
Tindices0*V
_classL
JHloc:@token_and_position_embedding_13/embedding_27/embedding_lookup/82773* 
_output_shapes
:
??*
dtype0?
Ftoken_and_position_embedding_13/embedding_27/embedding_lookup/IdentityIdentityFtoken_and_position_embedding_13/embedding_27/embedding_lookup:output:0*
T0*V
_classL
JHloc:@token_and_position_embedding_13/embedding_27/embedding_lookup/82773* 
_output_shapes
:
???
Htoken_and_position_embedding_13/embedding_27/embedding_lookup/Identity_1IdentityOtoken_and_position_embedding_13/embedding_27/embedding_lookup/Identity:output:0*
T0* 
_output_shapes
:
???
1token_and_position_embedding_13/embedding_26/CastCastinputs*

DstT0*

SrcT0*(
_output_shapes
:???????????
=token_and_position_embedding_13/embedding_26/embedding_lookupResourceGatherCtoken_and_position_embedding_13_embedding_26_embedding_lookup_827795token_and_position_embedding_13/embedding_26/Cast:y:0*
Tindices0*V
_classL
JHloc:@token_and_position_embedding_13/embedding_26/embedding_lookup/82779*-
_output_shapes
:???????????*
dtype0?
Ftoken_and_position_embedding_13/embedding_26/embedding_lookup/IdentityIdentityFtoken_and_position_embedding_13/embedding_26/embedding_lookup:output:0*
T0*V
_classL
JHloc:@token_and_position_embedding_13/embedding_26/embedding_lookup/82779*-
_output_shapes
:????????????
Htoken_and_position_embedding_13/embedding_26/embedding_lookup/Identity_1IdentityOtoken_and_position_embedding_13/embedding_26/embedding_lookup/Identity:output:0*
T0*-
_output_shapes
:????????????
#token_and_position_embedding_13/addAddV2Qtoken_and_position_embedding_13/embedding_26/embedding_lookup/Identity_1:output:0Qtoken_and_position_embedding_13/embedding_27/embedding_lookup/Identity_1:output:0*
T0*-
_output_shapes
:????????????
Otransformer_block_13/multi_head_attention_13/query/einsum/Einsum/ReadVariableOpReadVariableOpXtransformer_block_13_multi_head_attention_13_query_einsum_einsum_readvariableop_resource*$
_output_shapes
:??*
dtype0?
@transformer_block_13/multi_head_attention_13/query/einsum/EinsumEinsum'token_and_position_embedding_13/add:z:0Wtransformer_block_13/multi_head_attention_13/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*1
_output_shapes
:???????????*
equationabc,cde->abde?
Etransformer_block_13/multi_head_attention_13/query/add/ReadVariableOpReadVariableOpNtransformer_block_13_multi_head_attention_13_query_add_readvariableop_resource*
_output_shapes
:	?*
dtype0?
6transformer_block_13/multi_head_attention_13/query/addAddV2Itransformer_block_13/multi_head_attention_13/query/einsum/Einsum:output:0Mtransformer_block_13/multi_head_attention_13/query/add/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
Mtransformer_block_13/multi_head_attention_13/key/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_13_multi_head_attention_13_key_einsum_einsum_readvariableop_resource*$
_output_shapes
:??*
dtype0?
>transformer_block_13/multi_head_attention_13/key/einsum/EinsumEinsum'token_and_position_embedding_13/add:z:0Utransformer_block_13/multi_head_attention_13/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*1
_output_shapes
:???????????*
equationabc,cde->abde?
Ctransformer_block_13/multi_head_attention_13/key/add/ReadVariableOpReadVariableOpLtransformer_block_13_multi_head_attention_13_key_add_readvariableop_resource*
_output_shapes
:	?*
dtype0?
4transformer_block_13/multi_head_attention_13/key/addAddV2Gtransformer_block_13/multi_head_attention_13/key/einsum/Einsum:output:0Ktransformer_block_13/multi_head_attention_13/key/add/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
Otransformer_block_13/multi_head_attention_13/value/einsum/Einsum/ReadVariableOpReadVariableOpXtransformer_block_13_multi_head_attention_13_value_einsum_einsum_readvariableop_resource*$
_output_shapes
:??*
dtype0?
@transformer_block_13/multi_head_attention_13/value/einsum/EinsumEinsum'token_and_position_embedding_13/add:z:0Wtransformer_block_13/multi_head_attention_13/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*1
_output_shapes
:???????????*
equationabc,cde->abde?
Etransformer_block_13/multi_head_attention_13/value/add/ReadVariableOpReadVariableOpNtransformer_block_13_multi_head_attention_13_value_add_readvariableop_resource*
_output_shapes
:	?*
dtype0?
6transformer_block_13/multi_head_attention_13/value/addAddV2Itransformer_block_13/multi_head_attention_13/value/einsum/Einsum:output:0Mtransformer_block_13/multi_head_attention_13/value/add/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????w
2transformer_block_13/multi_head_attention_13/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *?А=?
0transformer_block_13/multi_head_attention_13/MulMul:transformer_block_13/multi_head_attention_13/query/add:z:0;transformer_block_13/multi_head_attention_13/Mul/y:output:0*
T0*1
_output_shapes
:????????????
:transformer_block_13/multi_head_attention_13/einsum/EinsumEinsum8transformer_block_13/multi_head_attention_13/key/add:z:04transformer_block_13/multi_head_attention_13/Mul:z:0*
N*
T0*1
_output_shapes
:???????????*
equationaecd,abcd->acbe?
<transformer_block_13/multi_head_attention_13/softmax/SoftmaxSoftmaxCtransformer_block_13/multi_head_attention_13/einsum/Einsum:output:0*
T0*1
_output_shapes
:????????????
<transformer_block_13/multi_head_attention_13/einsum_1/EinsumEinsumFtransformer_block_13/multi_head_attention_13/softmax/Softmax:softmax:0:transformer_block_13/multi_head_attention_13/value/add:z:0*
N*
T0*1
_output_shapes
:???????????*
equationacbe,aecd->abcd?
Ztransformer_block_13/multi_head_attention_13/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpctransformer_block_13_multi_head_attention_13_attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:??*
dtype0?
Ktransformer_block_13/multi_head_attention_13/attention_output/einsum/EinsumEinsumEtransformer_block_13/multi_head_attention_13/einsum_1/Einsum:output:0btransformer_block_13/multi_head_attention_13/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*-
_output_shapes
:???????????*
equationabcd,cde->abe?
Ptransformer_block_13/multi_head_attention_13/attention_output/add/ReadVariableOpReadVariableOpYtransformer_block_13_multi_head_attention_13_attention_output_add_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Atransformer_block_13/multi_head_attention_13/attention_output/addAddV2Ttransformer_block_13/multi_head_attention_13/attention_output/einsum/Einsum:output:0Xtransformer_block_13/multi_head_attention_13/attention_output/add/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????r
-transformer_block_13/dropout_55/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
+transformer_block_13/dropout_55/dropout/MulMulEtransformer_block_13/multi_head_attention_13/attention_output/add:z:06transformer_block_13/dropout_55/dropout/Const:output:0*
T0*-
_output_shapes
:????????????
-transformer_block_13/dropout_55/dropout/ShapeShapeEtransformer_block_13/multi_head_attention_13/attention_output/add:z:0*
T0*
_output_shapes
:?
Dtransformer_block_13/dropout_55/dropout/random_uniform/RandomUniformRandomUniform6transformer_block_13/dropout_55/dropout/Shape:output:0*
T0*-
_output_shapes
:???????????*
dtype0{
6transformer_block_13/dropout_55/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
4transformer_block_13/dropout_55/dropout/GreaterEqualGreaterEqualMtransformer_block_13/dropout_55/dropout/random_uniform/RandomUniform:output:0?transformer_block_13/dropout_55/dropout/GreaterEqual/y:output:0*
T0*-
_output_shapes
:????????????
,transformer_block_13/dropout_55/dropout/CastCast8transformer_block_13/dropout_55/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*-
_output_shapes
:????????????
-transformer_block_13/dropout_55/dropout/Mul_1Mul/transformer_block_13/dropout_55/dropout/Mul:z:00transformer_block_13/dropout_55/dropout/Cast:y:0*
T0*-
_output_shapes
:????????????
transformer_block_13/addAddV2'token_and_position_embedding_13/add:z:01transformer_block_13/dropout_55/dropout/Mul_1:z:0*
T0*-
_output_shapes
:???????????}
1transformer_block_13/layer_normalization_26/ShapeShapetransformer_block_13/add:z:0*
T0*
_output_shapes
:?
?transformer_block_13/layer_normalization_26/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Atransformer_block_13/layer_normalization_26/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Atransformer_block_13/layer_normalization_26/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
9transformer_block_13/layer_normalization_26/strided_sliceStridedSlice:transformer_block_13/layer_normalization_26/Shape:output:0Htransformer_block_13/layer_normalization_26/strided_slice/stack:output:0Jtransformer_block_13/layer_normalization_26/strided_slice/stack_1:output:0Jtransformer_block_13/layer_normalization_26/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
1transformer_block_13/layer_normalization_26/mul/xConst*
_output_shapes
: *
dtype0*
value	B :?
/transformer_block_13/layer_normalization_26/mulMul:transformer_block_13/layer_normalization_26/mul/x:output:0Btransformer_block_13/layer_normalization_26/strided_slice:output:0*
T0*
_output_shapes
: ?
Atransformer_block_13/layer_normalization_26/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Ctransformer_block_13/layer_normalization_26/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Ctransformer_block_13/layer_normalization_26/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;transformer_block_13/layer_normalization_26/strided_slice_1StridedSlice:transformer_block_13/layer_normalization_26/Shape:output:0Jtransformer_block_13/layer_normalization_26/strided_slice_1/stack:output:0Ltransformer_block_13/layer_normalization_26/strided_slice_1/stack_1:output:0Ltransformer_block_13/layer_normalization_26/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
1transformer_block_13/layer_normalization_26/mul_1Mul3transformer_block_13/layer_normalization_26/mul:z:0Dtransformer_block_13/layer_normalization_26/strided_slice_1:output:0*
T0*
_output_shapes
: ?
Atransformer_block_13/layer_normalization_26/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Ctransformer_block_13/layer_normalization_26/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Ctransformer_block_13/layer_normalization_26/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;transformer_block_13/layer_normalization_26/strided_slice_2StridedSlice:transformer_block_13/layer_normalization_26/Shape:output:0Jtransformer_block_13/layer_normalization_26/strided_slice_2/stack:output:0Ltransformer_block_13/layer_normalization_26/strided_slice_2/stack_1:output:0Ltransformer_block_13/layer_normalization_26/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3transformer_block_13/layer_normalization_26/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :?
1transformer_block_13/layer_normalization_26/mul_2Mul<transformer_block_13/layer_normalization_26/mul_2/x:output:0Dtransformer_block_13/layer_normalization_26/strided_slice_2:output:0*
T0*
_output_shapes
: }
;transformer_block_13/layer_normalization_26/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :}
;transformer_block_13/layer_normalization_26/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
9transformer_block_13/layer_normalization_26/Reshape/shapePackDtransformer_block_13/layer_normalization_26/Reshape/shape/0:output:05transformer_block_13/layer_normalization_26/mul_1:z:05transformer_block_13/layer_normalization_26/mul_2:z:0Dtransformer_block_13/layer_normalization_26/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
3transformer_block_13/layer_normalization_26/ReshapeReshapetransformer_block_13/add:z:0Btransformer_block_13/layer_normalization_26/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????z
7transformer_block_13/layer_normalization_26/ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :??
5transformer_block_13/layer_normalization_26/ones/LessLess5transformer_block_13/layer_normalization_26/mul_1:z:0@transformer_block_13/layer_normalization_26/ones/Less/y:output:0*
T0*
_output_shapes
: ?
7transformer_block_13/layer_normalization_26/ones/packedPack5transformer_block_13/layer_normalization_26/mul_1:z:0*
N*
T0*
_output_shapes
:{
6transformer_block_13/layer_normalization_26/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
0transformer_block_13/layer_normalization_26/onesFill@transformer_block_13/layer_normalization_26/ones/packed:output:0?transformer_block_13/layer_normalization_26/ones/Const:output:0*
T0*#
_output_shapes
:?????????{
8transformer_block_13/layer_normalization_26/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :??
6transformer_block_13/layer_normalization_26/zeros/LessLess5transformer_block_13/layer_normalization_26/mul_1:z:0Atransformer_block_13/layer_normalization_26/zeros/Less/y:output:0*
T0*
_output_shapes
: ?
8transformer_block_13/layer_normalization_26/zeros/packedPack5transformer_block_13/layer_normalization_26/mul_1:z:0*
N*
T0*
_output_shapes
:|
7transformer_block_13/layer_normalization_26/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
1transformer_block_13/layer_normalization_26/zerosFillAtransformer_block_13/layer_normalization_26/zeros/packed:output:0@transformer_block_13/layer_normalization_26/zeros/Const:output:0*
T0*#
_output_shapes
:?????????t
1transformer_block_13/layer_normalization_26/ConstConst*
_output_shapes
: *
dtype0*
valueB v
3transformer_block_13/layer_normalization_26/Const_1Const*
_output_shapes
: *
dtype0*
valueB ?
<transformer_block_13/layer_normalization_26/FusedBatchNormV3FusedBatchNormV3<transformer_block_13/layer_normalization_26/Reshape:output:09transformer_block_13/layer_normalization_26/ones:output:0:transformer_block_13/layer_normalization_26/zeros:output:0:transformer_block_13/layer_normalization_26/Const:output:0<transformer_block_13/layer_normalization_26/Const_1:output:0*
T0*
U0*p
_output_shapes^
\:??????????:?????????:?????????:?????????:?????????:*
data_formatNCHW?
5transformer_block_13/layer_normalization_26/Reshape_1Reshape@transformer_block_13/layer_normalization_26/FusedBatchNormV3:y:0:transformer_block_13/layer_normalization_26/Shape:output:0*
T0*-
_output_shapes
:????????????
@transformer_block_13/layer_normalization_26/mul_3/ReadVariableOpReadVariableOpItransformer_block_13_layer_normalization_26_mul_3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
1transformer_block_13/layer_normalization_26/mul_3Mul>transformer_block_13/layer_normalization_26/Reshape_1:output:0Htransformer_block_13/layer_normalization_26/mul_3/ReadVariableOp:value:0*
T0*-
_output_shapes
:????????????
>transformer_block_13/layer_normalization_26/add/ReadVariableOpReadVariableOpGtransformer_block_13_layer_normalization_26_add_readvariableop_resource*
_output_shapes	
:?*
dtype0?
/transformer_block_13/layer_normalization_26/addAddV25transformer_block_13/layer_normalization_26/mul_3:z:0Ftransformer_block_13/layer_normalization_26/add/ReadVariableOp:value:0*
T0*-
_output_shapes
:????????????
Dtransformer_block_13/sequential_13/dense_52/Tensordot/ReadVariableOpReadVariableOpMtransformer_block_13_sequential_13_dense_52_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
:transformer_block_13/sequential_13/dense_52/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
:transformer_block_13/sequential_13/dense_52/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
;transformer_block_13/sequential_13/dense_52/Tensordot/ShapeShape3transformer_block_13/layer_normalization_26/add:z:0*
T0*
_output_shapes
:?
Ctransformer_block_13/sequential_13/dense_52/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
>transformer_block_13/sequential_13/dense_52/Tensordot/GatherV2GatherV2Dtransformer_block_13/sequential_13/dense_52/Tensordot/Shape:output:0Ctransformer_block_13/sequential_13/dense_52/Tensordot/free:output:0Ltransformer_block_13/sequential_13/dense_52/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Etransformer_block_13/sequential_13/dense_52/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
@transformer_block_13/sequential_13/dense_52/Tensordot/GatherV2_1GatherV2Dtransformer_block_13/sequential_13/dense_52/Tensordot/Shape:output:0Ctransformer_block_13/sequential_13/dense_52/Tensordot/axes:output:0Ntransformer_block_13/sequential_13/dense_52/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
;transformer_block_13/sequential_13/dense_52/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
:transformer_block_13/sequential_13/dense_52/Tensordot/ProdProdGtransformer_block_13/sequential_13/dense_52/Tensordot/GatherV2:output:0Dtransformer_block_13/sequential_13/dense_52/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
=transformer_block_13/sequential_13/dense_52/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
<transformer_block_13/sequential_13/dense_52/Tensordot/Prod_1ProdItransformer_block_13/sequential_13/dense_52/Tensordot/GatherV2_1:output:0Ftransformer_block_13/sequential_13/dense_52/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ?
Atransformer_block_13/sequential_13/dense_52/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
<transformer_block_13/sequential_13/dense_52/Tensordot/concatConcatV2Ctransformer_block_13/sequential_13/dense_52/Tensordot/free:output:0Ctransformer_block_13/sequential_13/dense_52/Tensordot/axes:output:0Jtransformer_block_13/sequential_13/dense_52/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
;transformer_block_13/sequential_13/dense_52/Tensordot/stackPackCtransformer_block_13/sequential_13/dense_52/Tensordot/Prod:output:0Etransformer_block_13/sequential_13/dense_52/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
?transformer_block_13/sequential_13/dense_52/Tensordot/transpose	Transpose3transformer_block_13/layer_normalization_26/add:z:0Etransformer_block_13/sequential_13/dense_52/Tensordot/concat:output:0*
T0*-
_output_shapes
:????????????
=transformer_block_13/sequential_13/dense_52/Tensordot/ReshapeReshapeCtransformer_block_13/sequential_13/dense_52/Tensordot/transpose:y:0Dtransformer_block_13/sequential_13/dense_52/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
<transformer_block_13/sequential_13/dense_52/Tensordot/MatMulMatMulFtransformer_block_13/sequential_13/dense_52/Tensordot/Reshape:output:0Ltransformer_block_13/sequential_13/dense_52/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
=transformer_block_13/sequential_13/dense_52/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:??
Ctransformer_block_13/sequential_13/dense_52/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
>transformer_block_13/sequential_13/dense_52/Tensordot/concat_1ConcatV2Gtransformer_block_13/sequential_13/dense_52/Tensordot/GatherV2:output:0Ftransformer_block_13/sequential_13/dense_52/Tensordot/Const_2:output:0Ltransformer_block_13/sequential_13/dense_52/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
5transformer_block_13/sequential_13/dense_52/TensordotReshapeFtransformer_block_13/sequential_13/dense_52/Tensordot/MatMul:product:0Gtransformer_block_13/sequential_13/dense_52/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:????????????
Btransformer_block_13/sequential_13/dense_52/BiasAdd/ReadVariableOpReadVariableOpKtransformer_block_13_sequential_13_dense_52_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
3transformer_block_13/sequential_13/dense_52/BiasAddBiasAdd>transformer_block_13/sequential_13/dense_52/Tensordot:output:0Jtransformer_block_13/sequential_13/dense_52/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:????????????
0transformer_block_13/sequential_13/dense_52/ReluRelu<transformer_block_13/sequential_13/dense_52/BiasAdd:output:0*
T0*-
_output_shapes
:????????????
Dtransformer_block_13/sequential_13/dense_53/Tensordot/ReadVariableOpReadVariableOpMtransformer_block_13_sequential_13_dense_53_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
:transformer_block_13/sequential_13/dense_53/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
:transformer_block_13/sequential_13/dense_53/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
;transformer_block_13/sequential_13/dense_53/Tensordot/ShapeShape>transformer_block_13/sequential_13/dense_52/Relu:activations:0*
T0*
_output_shapes
:?
Ctransformer_block_13/sequential_13/dense_53/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
>transformer_block_13/sequential_13/dense_53/Tensordot/GatherV2GatherV2Dtransformer_block_13/sequential_13/dense_53/Tensordot/Shape:output:0Ctransformer_block_13/sequential_13/dense_53/Tensordot/free:output:0Ltransformer_block_13/sequential_13/dense_53/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Etransformer_block_13/sequential_13/dense_53/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
@transformer_block_13/sequential_13/dense_53/Tensordot/GatherV2_1GatherV2Dtransformer_block_13/sequential_13/dense_53/Tensordot/Shape:output:0Ctransformer_block_13/sequential_13/dense_53/Tensordot/axes:output:0Ntransformer_block_13/sequential_13/dense_53/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
;transformer_block_13/sequential_13/dense_53/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
:transformer_block_13/sequential_13/dense_53/Tensordot/ProdProdGtransformer_block_13/sequential_13/dense_53/Tensordot/GatherV2:output:0Dtransformer_block_13/sequential_13/dense_53/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
=transformer_block_13/sequential_13/dense_53/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
<transformer_block_13/sequential_13/dense_53/Tensordot/Prod_1ProdItransformer_block_13/sequential_13/dense_53/Tensordot/GatherV2_1:output:0Ftransformer_block_13/sequential_13/dense_53/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ?
Atransformer_block_13/sequential_13/dense_53/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
<transformer_block_13/sequential_13/dense_53/Tensordot/concatConcatV2Ctransformer_block_13/sequential_13/dense_53/Tensordot/free:output:0Ctransformer_block_13/sequential_13/dense_53/Tensordot/axes:output:0Jtransformer_block_13/sequential_13/dense_53/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
;transformer_block_13/sequential_13/dense_53/Tensordot/stackPackCtransformer_block_13/sequential_13/dense_53/Tensordot/Prod:output:0Etransformer_block_13/sequential_13/dense_53/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
?transformer_block_13/sequential_13/dense_53/Tensordot/transpose	Transpose>transformer_block_13/sequential_13/dense_52/Relu:activations:0Etransformer_block_13/sequential_13/dense_53/Tensordot/concat:output:0*
T0*-
_output_shapes
:????????????
=transformer_block_13/sequential_13/dense_53/Tensordot/ReshapeReshapeCtransformer_block_13/sequential_13/dense_53/Tensordot/transpose:y:0Dtransformer_block_13/sequential_13/dense_53/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
<transformer_block_13/sequential_13/dense_53/Tensordot/MatMulMatMulFtransformer_block_13/sequential_13/dense_53/Tensordot/Reshape:output:0Ltransformer_block_13/sequential_13/dense_53/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
=transformer_block_13/sequential_13/dense_53/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:??
Ctransformer_block_13/sequential_13/dense_53/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
>transformer_block_13/sequential_13/dense_53/Tensordot/concat_1ConcatV2Gtransformer_block_13/sequential_13/dense_53/Tensordot/GatherV2:output:0Ftransformer_block_13/sequential_13/dense_53/Tensordot/Const_2:output:0Ltransformer_block_13/sequential_13/dense_53/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
5transformer_block_13/sequential_13/dense_53/TensordotReshapeFtransformer_block_13/sequential_13/dense_53/Tensordot/MatMul:product:0Gtransformer_block_13/sequential_13/dense_53/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:????????????
Btransformer_block_13/sequential_13/dense_53/BiasAdd/ReadVariableOpReadVariableOpKtransformer_block_13_sequential_13_dense_53_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
3transformer_block_13/sequential_13/dense_53/BiasAddBiasAdd>transformer_block_13/sequential_13/dense_53/Tensordot:output:0Jtransformer_block_13/sequential_13/dense_53/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????r
-transformer_block_13/dropout_56/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
+transformer_block_13/dropout_56/dropout/MulMul<transformer_block_13/sequential_13/dense_53/BiasAdd:output:06transformer_block_13/dropout_56/dropout/Const:output:0*
T0*-
_output_shapes
:????????????
-transformer_block_13/dropout_56/dropout/ShapeShape<transformer_block_13/sequential_13/dense_53/BiasAdd:output:0*
T0*
_output_shapes
:?
Dtransformer_block_13/dropout_56/dropout/random_uniform/RandomUniformRandomUniform6transformer_block_13/dropout_56/dropout/Shape:output:0*
T0*-
_output_shapes
:???????????*
dtype0{
6transformer_block_13/dropout_56/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
4transformer_block_13/dropout_56/dropout/GreaterEqualGreaterEqualMtransformer_block_13/dropout_56/dropout/random_uniform/RandomUniform:output:0?transformer_block_13/dropout_56/dropout/GreaterEqual/y:output:0*
T0*-
_output_shapes
:????????????
,transformer_block_13/dropout_56/dropout/CastCast8transformer_block_13/dropout_56/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*-
_output_shapes
:????????????
-transformer_block_13/dropout_56/dropout/Mul_1Mul/transformer_block_13/dropout_56/dropout/Mul:z:00transformer_block_13/dropout_56/dropout/Cast:y:0*
T0*-
_output_shapes
:????????????
transformer_block_13/add_1AddV23transformer_block_13/layer_normalization_26/add:z:01transformer_block_13/dropout_56/dropout/Mul_1:z:0*
T0*-
_output_shapes
:???????????
1transformer_block_13/layer_normalization_27/ShapeShapetransformer_block_13/add_1:z:0*
T0*
_output_shapes
:?
?transformer_block_13/layer_normalization_27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Atransformer_block_13/layer_normalization_27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Atransformer_block_13/layer_normalization_27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
9transformer_block_13/layer_normalization_27/strided_sliceStridedSlice:transformer_block_13/layer_normalization_27/Shape:output:0Htransformer_block_13/layer_normalization_27/strided_slice/stack:output:0Jtransformer_block_13/layer_normalization_27/strided_slice/stack_1:output:0Jtransformer_block_13/layer_normalization_27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
1transformer_block_13/layer_normalization_27/mul/xConst*
_output_shapes
: *
dtype0*
value	B :?
/transformer_block_13/layer_normalization_27/mulMul:transformer_block_13/layer_normalization_27/mul/x:output:0Btransformer_block_13/layer_normalization_27/strided_slice:output:0*
T0*
_output_shapes
: ?
Atransformer_block_13/layer_normalization_27/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Ctransformer_block_13/layer_normalization_27/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Ctransformer_block_13/layer_normalization_27/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;transformer_block_13/layer_normalization_27/strided_slice_1StridedSlice:transformer_block_13/layer_normalization_27/Shape:output:0Jtransformer_block_13/layer_normalization_27/strided_slice_1/stack:output:0Ltransformer_block_13/layer_normalization_27/strided_slice_1/stack_1:output:0Ltransformer_block_13/layer_normalization_27/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
1transformer_block_13/layer_normalization_27/mul_1Mul3transformer_block_13/layer_normalization_27/mul:z:0Dtransformer_block_13/layer_normalization_27/strided_slice_1:output:0*
T0*
_output_shapes
: ?
Atransformer_block_13/layer_normalization_27/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Ctransformer_block_13/layer_normalization_27/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Ctransformer_block_13/layer_normalization_27/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;transformer_block_13/layer_normalization_27/strided_slice_2StridedSlice:transformer_block_13/layer_normalization_27/Shape:output:0Jtransformer_block_13/layer_normalization_27/strided_slice_2/stack:output:0Ltransformer_block_13/layer_normalization_27/strided_slice_2/stack_1:output:0Ltransformer_block_13/layer_normalization_27/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3transformer_block_13/layer_normalization_27/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :?
1transformer_block_13/layer_normalization_27/mul_2Mul<transformer_block_13/layer_normalization_27/mul_2/x:output:0Dtransformer_block_13/layer_normalization_27/strided_slice_2:output:0*
T0*
_output_shapes
: }
;transformer_block_13/layer_normalization_27/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :}
;transformer_block_13/layer_normalization_27/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
9transformer_block_13/layer_normalization_27/Reshape/shapePackDtransformer_block_13/layer_normalization_27/Reshape/shape/0:output:05transformer_block_13/layer_normalization_27/mul_1:z:05transformer_block_13/layer_normalization_27/mul_2:z:0Dtransformer_block_13/layer_normalization_27/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
3transformer_block_13/layer_normalization_27/ReshapeReshapetransformer_block_13/add_1:z:0Btransformer_block_13/layer_normalization_27/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????z
7transformer_block_13/layer_normalization_27/ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :??
5transformer_block_13/layer_normalization_27/ones/LessLess5transformer_block_13/layer_normalization_27/mul_1:z:0@transformer_block_13/layer_normalization_27/ones/Less/y:output:0*
T0*
_output_shapes
: ?
7transformer_block_13/layer_normalization_27/ones/packedPack5transformer_block_13/layer_normalization_27/mul_1:z:0*
N*
T0*
_output_shapes
:{
6transformer_block_13/layer_normalization_27/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
0transformer_block_13/layer_normalization_27/onesFill@transformer_block_13/layer_normalization_27/ones/packed:output:0?transformer_block_13/layer_normalization_27/ones/Const:output:0*
T0*#
_output_shapes
:?????????{
8transformer_block_13/layer_normalization_27/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :??
6transformer_block_13/layer_normalization_27/zeros/LessLess5transformer_block_13/layer_normalization_27/mul_1:z:0Atransformer_block_13/layer_normalization_27/zeros/Less/y:output:0*
T0*
_output_shapes
: ?
8transformer_block_13/layer_normalization_27/zeros/packedPack5transformer_block_13/layer_normalization_27/mul_1:z:0*
N*
T0*
_output_shapes
:|
7transformer_block_13/layer_normalization_27/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
1transformer_block_13/layer_normalization_27/zerosFillAtransformer_block_13/layer_normalization_27/zeros/packed:output:0@transformer_block_13/layer_normalization_27/zeros/Const:output:0*
T0*#
_output_shapes
:?????????t
1transformer_block_13/layer_normalization_27/ConstConst*
_output_shapes
: *
dtype0*
valueB v
3transformer_block_13/layer_normalization_27/Const_1Const*
_output_shapes
: *
dtype0*
valueB ?
<transformer_block_13/layer_normalization_27/FusedBatchNormV3FusedBatchNormV3<transformer_block_13/layer_normalization_27/Reshape:output:09transformer_block_13/layer_normalization_27/ones:output:0:transformer_block_13/layer_normalization_27/zeros:output:0:transformer_block_13/layer_normalization_27/Const:output:0<transformer_block_13/layer_normalization_27/Const_1:output:0*
T0*
U0*p
_output_shapes^
\:??????????:?????????:?????????:?????????:?????????:*
data_formatNCHW?
5transformer_block_13/layer_normalization_27/Reshape_1Reshape@transformer_block_13/layer_normalization_27/FusedBatchNormV3:y:0:transformer_block_13/layer_normalization_27/Shape:output:0*
T0*-
_output_shapes
:????????????
@transformer_block_13/layer_normalization_27/mul_3/ReadVariableOpReadVariableOpItransformer_block_13_layer_normalization_27_mul_3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
1transformer_block_13/layer_normalization_27/mul_3Mul>transformer_block_13/layer_normalization_27/Reshape_1:output:0Htransformer_block_13/layer_normalization_27/mul_3/ReadVariableOp:value:0*
T0*-
_output_shapes
:????????????
>transformer_block_13/layer_normalization_27/add/ReadVariableOpReadVariableOpGtransformer_block_13_layer_normalization_27_add_readvariableop_resource*
_output_shapes	
:?*
dtype0?
/transformer_block_13/layer_normalization_27/addAddV25transformer_block_13/layer_normalization_27/mul_3:z:0Ftransformer_block_13/layer_normalization_27/add/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????t
2global_average_pooling1d_13/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
 global_average_pooling1d_13/MeanMean3transformer_block_13/layer_normalization_27/add:z:0;global_average_pooling1d_13/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:??????????]
dropout_57/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout_57/dropout/MulMul)global_average_pooling1d_13/Mean:output:0!dropout_57/dropout/Const:output:0*
T0*(
_output_shapes
:??????????q
dropout_57/dropout/ShapeShape)global_average_pooling1d_13/Mean:output:0*
T0*
_output_shapes
:?
/dropout_57/dropout/random_uniform/RandomUniformRandomUniform!dropout_57/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0f
!dropout_57/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout_57/dropout/GreaterEqualGreaterEqual8dropout_57/dropout/random_uniform/RandomUniform:output:0*dropout_57/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
dropout_57/dropout/CastCast#dropout_57/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
dropout_57/dropout/Mul_1Muldropout_57/dropout/Mul:z:0dropout_57/dropout/Cast:y:0*
T0*(
_output_shapes
:???????????
dense_54/MatMul/ReadVariableOpReadVariableOp'dense_54_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense_54/MatMulMatMuldropout_57/dropout/Mul_1:z:0&dense_54/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_54/BiasAdd/ReadVariableOpReadVariableOp(dense_54_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_54/BiasAddBiasAdddense_54/MatMul:product:0'dense_54/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????b
dense_54/ReluReludense_54/BiasAdd:output:0*
T0*'
_output_shapes
:?????????]
dropout_58/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout_58/dropout/MulMuldense_54/Relu:activations:0!dropout_58/dropout/Const:output:0*
T0*'
_output_shapes
:?????????c
dropout_58/dropout/ShapeShapedense_54/Relu:activations:0*
T0*
_output_shapes
:?
/dropout_58/dropout/random_uniform/RandomUniformRandomUniform!dropout_58/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0f
!dropout_58/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout_58/dropout/GreaterEqualGreaterEqual8dropout_58/dropout/random_uniform/RandomUniform:output:0*dropout_58/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
dropout_58/dropout/CastCast#dropout_58/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
dropout_58/dropout/Mul_1Muldropout_58/dropout/Mul:z:0dropout_58/dropout/Cast:y:0*
T0*'
_output_shapes
:??????????
dense_55/MatMul/ReadVariableOpReadVariableOp'dense_55_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_55/MatMulMatMuldropout_58/dropout/Mul_1:z:0&dense_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_55/BiasAdd/ReadVariableOpReadVariableOp(dense_55_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_55/BiasAddBiasAdddense_55/MatMul:product:0'dense_55/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
dense_55/SoftmaxSoftmaxdense_55/BiasAdd:output:0*
T0*'
_output_shapes
:?????????i
IdentityIdentitydense_55/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_54/BiasAdd/ReadVariableOp^dense_54/MatMul/ReadVariableOp ^dense_55/BiasAdd/ReadVariableOp^dense_55/MatMul/ReadVariableOp>^token_and_position_embedding_13/embedding_26/embedding_lookup>^token_and_position_embedding_13/embedding_27/embedding_lookup?^transformer_block_13/layer_normalization_26/add/ReadVariableOpA^transformer_block_13/layer_normalization_26/mul_3/ReadVariableOp?^transformer_block_13/layer_normalization_27/add/ReadVariableOpA^transformer_block_13/layer_normalization_27/mul_3/ReadVariableOpQ^transformer_block_13/multi_head_attention_13/attention_output/add/ReadVariableOp[^transformer_block_13/multi_head_attention_13/attention_output/einsum/Einsum/ReadVariableOpD^transformer_block_13/multi_head_attention_13/key/add/ReadVariableOpN^transformer_block_13/multi_head_attention_13/key/einsum/Einsum/ReadVariableOpF^transformer_block_13/multi_head_attention_13/query/add/ReadVariableOpP^transformer_block_13/multi_head_attention_13/query/einsum/Einsum/ReadVariableOpF^transformer_block_13/multi_head_attention_13/value/add/ReadVariableOpP^transformer_block_13/multi_head_attention_13/value/einsum/Einsum/ReadVariableOpC^transformer_block_13/sequential_13/dense_52/BiasAdd/ReadVariableOpE^transformer_block_13/sequential_13/dense_52/Tensordot/ReadVariableOpC^transformer_block_13/sequential_13/dense_53/BiasAdd/ReadVariableOpE^transformer_block_13/sequential_13/dense_53/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:??????????: : : : : : : : : : : : : : : : : : : : : : 2B
dense_54/BiasAdd/ReadVariableOpdense_54/BiasAdd/ReadVariableOp2@
dense_54/MatMul/ReadVariableOpdense_54/MatMul/ReadVariableOp2B
dense_55/BiasAdd/ReadVariableOpdense_55/BiasAdd/ReadVariableOp2@
dense_55/MatMul/ReadVariableOpdense_55/MatMul/ReadVariableOp2~
=token_and_position_embedding_13/embedding_26/embedding_lookup=token_and_position_embedding_13/embedding_26/embedding_lookup2~
=token_and_position_embedding_13/embedding_27/embedding_lookup=token_and_position_embedding_13/embedding_27/embedding_lookup2?
>transformer_block_13/layer_normalization_26/add/ReadVariableOp>transformer_block_13/layer_normalization_26/add/ReadVariableOp2?
@transformer_block_13/layer_normalization_26/mul_3/ReadVariableOp@transformer_block_13/layer_normalization_26/mul_3/ReadVariableOp2?
>transformer_block_13/layer_normalization_27/add/ReadVariableOp>transformer_block_13/layer_normalization_27/add/ReadVariableOp2?
@transformer_block_13/layer_normalization_27/mul_3/ReadVariableOp@transformer_block_13/layer_normalization_27/mul_3/ReadVariableOp2?
Ptransformer_block_13/multi_head_attention_13/attention_output/add/ReadVariableOpPtransformer_block_13/multi_head_attention_13/attention_output/add/ReadVariableOp2?
Ztransformer_block_13/multi_head_attention_13/attention_output/einsum/Einsum/ReadVariableOpZtransformer_block_13/multi_head_attention_13/attention_output/einsum/Einsum/ReadVariableOp2?
Ctransformer_block_13/multi_head_attention_13/key/add/ReadVariableOpCtransformer_block_13/multi_head_attention_13/key/add/ReadVariableOp2?
Mtransformer_block_13/multi_head_attention_13/key/einsum/Einsum/ReadVariableOpMtransformer_block_13/multi_head_attention_13/key/einsum/Einsum/ReadVariableOp2?
Etransformer_block_13/multi_head_attention_13/query/add/ReadVariableOpEtransformer_block_13/multi_head_attention_13/query/add/ReadVariableOp2?
Otransformer_block_13/multi_head_attention_13/query/einsum/Einsum/ReadVariableOpOtransformer_block_13/multi_head_attention_13/query/einsum/Einsum/ReadVariableOp2?
Etransformer_block_13/multi_head_attention_13/value/add/ReadVariableOpEtransformer_block_13/multi_head_attention_13/value/add/ReadVariableOp2?
Otransformer_block_13/multi_head_attention_13/value/einsum/Einsum/ReadVariableOpOtransformer_block_13/multi_head_attention_13/value/einsum/Einsum/ReadVariableOp2?
Btransformer_block_13/sequential_13/dense_52/BiasAdd/ReadVariableOpBtransformer_block_13/sequential_13/dense_52/BiasAdd/ReadVariableOp2?
Dtransformer_block_13/sequential_13/dense_52/Tensordot/ReadVariableOpDtransformer_block_13/sequential_13/dense_52/Tensordot/ReadVariableOp2?
Btransformer_block_13/sequential_13/dense_53/BiasAdd/ReadVariableOpBtransformer_block_13/sequential_13/dense_53/BiasAdd/ReadVariableOp2?
Dtransformer_block_13/sequential_13/dense_53/Tensordot/ReadVariableOpDtransformer_block_13/sequential_13/dense_53/Tensordot/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
?__inference_token_and_position_embedding_13_layer_call_fn_83021
x
unknown:
??
	unknown_0:
?)?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *c
f^R\
Z__inference_token_and_position_embedding_13_layer_call_and_return_conditional_losses_81390u
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:??????????

_user_specified_namex
??
?
O__inference_transformer_block_13_layer_call_and_return_conditional_losses_82027

inputs[
Cmulti_head_attention_13_query_einsum_einsum_readvariableop_resource:??L
9multi_head_attention_13_query_add_readvariableop_resource:	?Y
Amulti_head_attention_13_key_einsum_einsum_readvariableop_resource:??J
7multi_head_attention_13_key_add_readvariableop_resource:	?[
Cmulti_head_attention_13_value_einsum_einsum_readvariableop_resource:??L
9multi_head_attention_13_value_add_readvariableop_resource:	?f
Nmulti_head_attention_13_attention_output_einsum_einsum_readvariableop_resource:??S
Dmulti_head_attention_13_attention_output_add_readvariableop_resource:	?C
4layer_normalization_26_mul_3_readvariableop_resource:	?A
2layer_normalization_26_add_readvariableop_resource:	?L
8sequential_13_dense_52_tensordot_readvariableop_resource:
??E
6sequential_13_dense_52_biasadd_readvariableop_resource:	?L
8sequential_13_dense_53_tensordot_readvariableop_resource:
??E
6sequential_13_dense_53_biasadd_readvariableop_resource:	?C
4layer_normalization_27_mul_3_readvariableop_resource:	?A
2layer_normalization_27_add_readvariableop_resource:	?
identity??)layer_normalization_26/add/ReadVariableOp?+layer_normalization_26/mul_3/ReadVariableOp?)layer_normalization_27/add/ReadVariableOp?+layer_normalization_27/mul_3/ReadVariableOp?;multi_head_attention_13/attention_output/add/ReadVariableOp?Emulti_head_attention_13/attention_output/einsum/Einsum/ReadVariableOp?.multi_head_attention_13/key/add/ReadVariableOp?8multi_head_attention_13/key/einsum/Einsum/ReadVariableOp?0multi_head_attention_13/query/add/ReadVariableOp?:multi_head_attention_13/query/einsum/Einsum/ReadVariableOp?0multi_head_attention_13/value/add/ReadVariableOp?:multi_head_attention_13/value/einsum/Einsum/ReadVariableOp?-sequential_13/dense_52/BiasAdd/ReadVariableOp?/sequential_13/dense_52/Tensordot/ReadVariableOp?-sequential_13/dense_53/BiasAdd/ReadVariableOp?/sequential_13/dense_53/Tensordot/ReadVariableOp?
:multi_head_attention_13/query/einsum/Einsum/ReadVariableOpReadVariableOpCmulti_head_attention_13_query_einsum_einsum_readvariableop_resource*$
_output_shapes
:??*
dtype0?
+multi_head_attention_13/query/einsum/EinsumEinsuminputsBmulti_head_attention_13/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*1
_output_shapes
:???????????*
equationabc,cde->abde?
0multi_head_attention_13/query/add/ReadVariableOpReadVariableOp9multi_head_attention_13_query_add_readvariableop_resource*
_output_shapes
:	?*
dtype0?
!multi_head_attention_13/query/addAddV24multi_head_attention_13/query/einsum/Einsum:output:08multi_head_attention_13/query/add/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
8multi_head_attention_13/key/einsum/Einsum/ReadVariableOpReadVariableOpAmulti_head_attention_13_key_einsum_einsum_readvariableop_resource*$
_output_shapes
:??*
dtype0?
)multi_head_attention_13/key/einsum/EinsumEinsuminputs@multi_head_attention_13/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*1
_output_shapes
:???????????*
equationabc,cde->abde?
.multi_head_attention_13/key/add/ReadVariableOpReadVariableOp7multi_head_attention_13_key_add_readvariableop_resource*
_output_shapes
:	?*
dtype0?
multi_head_attention_13/key/addAddV22multi_head_attention_13/key/einsum/Einsum:output:06multi_head_attention_13/key/add/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
:multi_head_attention_13/value/einsum/Einsum/ReadVariableOpReadVariableOpCmulti_head_attention_13_value_einsum_einsum_readvariableop_resource*$
_output_shapes
:??*
dtype0?
+multi_head_attention_13/value/einsum/EinsumEinsuminputsBmulti_head_attention_13/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*1
_output_shapes
:???????????*
equationabc,cde->abde?
0multi_head_attention_13/value/add/ReadVariableOpReadVariableOp9multi_head_attention_13_value_add_readvariableop_resource*
_output_shapes
:	?*
dtype0?
!multi_head_attention_13/value/addAddV24multi_head_attention_13/value/einsum/Einsum:output:08multi_head_attention_13/value/add/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????b
multi_head_attention_13/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *?А=?
multi_head_attention_13/MulMul%multi_head_attention_13/query/add:z:0&multi_head_attention_13/Mul/y:output:0*
T0*1
_output_shapes
:????????????
%multi_head_attention_13/einsum/EinsumEinsum#multi_head_attention_13/key/add:z:0multi_head_attention_13/Mul:z:0*
N*
T0*1
_output_shapes
:???????????*
equationaecd,abcd->acbe?
'multi_head_attention_13/softmax/SoftmaxSoftmax.multi_head_attention_13/einsum/Einsum:output:0*
T0*1
_output_shapes
:????????????
'multi_head_attention_13/einsum_1/EinsumEinsum1multi_head_attention_13/softmax/Softmax:softmax:0%multi_head_attention_13/value/add:z:0*
N*
T0*1
_output_shapes
:???????????*
equationacbe,aecd->abcd?
Emulti_head_attention_13/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpNmulti_head_attention_13_attention_output_einsum_einsum_readvariableop_resource*$
_output_shapes
:??*
dtype0?
6multi_head_attention_13/attention_output/einsum/EinsumEinsum0multi_head_attention_13/einsum_1/Einsum:output:0Mmulti_head_attention_13/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*-
_output_shapes
:???????????*
equationabcd,cde->abe?
;multi_head_attention_13/attention_output/add/ReadVariableOpReadVariableOpDmulti_head_attention_13_attention_output_add_readvariableop_resource*
_output_shapes	
:?*
dtype0?
,multi_head_attention_13/attention_output/addAddV2?multi_head_attention_13/attention_output/einsum/Einsum:output:0Cmulti_head_attention_13/attention_output/add/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????]
dropout_55/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout_55/dropout/MulMul0multi_head_attention_13/attention_output/add:z:0!dropout_55/dropout/Const:output:0*
T0*-
_output_shapes
:???????????x
dropout_55/dropout/ShapeShape0multi_head_attention_13/attention_output/add:z:0*
T0*
_output_shapes
:?
/dropout_55/dropout/random_uniform/RandomUniformRandomUniform!dropout_55/dropout/Shape:output:0*
T0*-
_output_shapes
:???????????*
dtype0f
!dropout_55/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout_55/dropout/GreaterEqualGreaterEqual8dropout_55/dropout/random_uniform/RandomUniform:output:0*dropout_55/dropout/GreaterEqual/y:output:0*
T0*-
_output_shapes
:????????????
dropout_55/dropout/CastCast#dropout_55/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*-
_output_shapes
:????????????
dropout_55/dropout/Mul_1Muldropout_55/dropout/Mul:z:0dropout_55/dropout/Cast:y:0*
T0*-
_output_shapes
:???????????j
addAddV2inputsdropout_55/dropout/Mul_1:z:0*
T0*-
_output_shapes
:???????????S
layer_normalization_26/ShapeShapeadd:z:0*
T0*
_output_shapes
:t
*layer_normalization_26/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,layer_normalization_26/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,layer_normalization_26/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$layer_normalization_26/strided_sliceStridedSlice%layer_normalization_26/Shape:output:03layer_normalization_26/strided_slice/stack:output:05layer_normalization_26/strided_slice/stack_1:output:05layer_normalization_26/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
layer_normalization_26/mul/xConst*
_output_shapes
: *
dtype0*
value	B :?
layer_normalization_26/mulMul%layer_normalization_26/mul/x:output:0-layer_normalization_26/strided_slice:output:0*
T0*
_output_shapes
: v
,layer_normalization_26/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_26/strided_slice_1StridedSlice%layer_normalization_26/Shape:output:05layer_normalization_26/strided_slice_1/stack:output:07layer_normalization_26/strided_slice_1/stack_1:output:07layer_normalization_26/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
layer_normalization_26/mul_1Mullayer_normalization_26/mul:z:0/layer_normalization_26/strided_slice_1:output:0*
T0*
_output_shapes
: v
,layer_normalization_26/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_26/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_26/strided_slice_2StridedSlice%layer_normalization_26/Shape:output:05layer_normalization_26/strided_slice_2/stack:output:07layer_normalization_26/strided_slice_2/stack_1:output:07layer_normalization_26/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
layer_normalization_26/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :?
layer_normalization_26/mul_2Mul'layer_normalization_26/mul_2/x:output:0/layer_normalization_26/strided_slice_2:output:0*
T0*
_output_shapes
: h
&layer_normalization_26/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :h
&layer_normalization_26/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
$layer_normalization_26/Reshape/shapePack/layer_normalization_26/Reshape/shape/0:output:0 layer_normalization_26/mul_1:z:0 layer_normalization_26/mul_2:z:0/layer_normalization_26/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
layer_normalization_26/ReshapeReshapeadd:z:0-layer_normalization_26/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????e
"layer_normalization_26/ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :??
 layer_normalization_26/ones/LessLess layer_normalization_26/mul_1:z:0+layer_normalization_26/ones/Less/y:output:0*
T0*
_output_shapes
: z
"layer_normalization_26/ones/packedPack layer_normalization_26/mul_1:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_26/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
layer_normalization_26/onesFill+layer_normalization_26/ones/packed:output:0*layer_normalization_26/ones/Const:output:0*
T0*#
_output_shapes
:?????????f
#layer_normalization_26/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :??
!layer_normalization_26/zeros/LessLess layer_normalization_26/mul_1:z:0,layer_normalization_26/zeros/Less/y:output:0*
T0*
_output_shapes
: {
#layer_normalization_26/zeros/packedPack layer_normalization_26/mul_1:z:0*
N*
T0*
_output_shapes
:g
"layer_normalization_26/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
layer_normalization_26/zerosFill,layer_normalization_26/zeros/packed:output:0+layer_normalization_26/zeros/Const:output:0*
T0*#
_output_shapes
:?????????_
layer_normalization_26/ConstConst*
_output_shapes
: *
dtype0*
valueB a
layer_normalization_26/Const_1Const*
_output_shapes
: *
dtype0*
valueB ?
'layer_normalization_26/FusedBatchNormV3FusedBatchNormV3'layer_normalization_26/Reshape:output:0$layer_normalization_26/ones:output:0%layer_normalization_26/zeros:output:0%layer_normalization_26/Const:output:0'layer_normalization_26/Const_1:output:0*
T0*
U0*p
_output_shapes^
\:??????????:?????????:?????????:?????????:?????????:*
data_formatNCHW?
 layer_normalization_26/Reshape_1Reshape+layer_normalization_26/FusedBatchNormV3:y:0%layer_normalization_26/Shape:output:0*
T0*-
_output_shapes
:????????????
+layer_normalization_26/mul_3/ReadVariableOpReadVariableOp4layer_normalization_26_mul_3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
layer_normalization_26/mul_3Mul)layer_normalization_26/Reshape_1:output:03layer_normalization_26/mul_3/ReadVariableOp:value:0*
T0*-
_output_shapes
:????????????
)layer_normalization_26/add/ReadVariableOpReadVariableOp2layer_normalization_26_add_readvariableop_resource*
_output_shapes	
:?*
dtype0?
layer_normalization_26/addAddV2 layer_normalization_26/mul_3:z:01layer_normalization_26/add/ReadVariableOp:value:0*
T0*-
_output_shapes
:????????????
/sequential_13/dense_52/Tensordot/ReadVariableOpReadVariableOp8sequential_13_dense_52_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype0o
%sequential_13/dense_52/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:v
%sequential_13/dense_52/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       t
&sequential_13/dense_52/Tensordot/ShapeShapelayer_normalization_26/add:z:0*
T0*
_output_shapes
:p
.sequential_13/dense_52/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
)sequential_13/dense_52/Tensordot/GatherV2GatherV2/sequential_13/dense_52/Tensordot/Shape:output:0.sequential_13/dense_52/Tensordot/free:output:07sequential_13/dense_52/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:r
0sequential_13/dense_52/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+sequential_13/dense_52/Tensordot/GatherV2_1GatherV2/sequential_13/dense_52/Tensordot/Shape:output:0.sequential_13/dense_52/Tensordot/axes:output:09sequential_13/dense_52/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
&sequential_13/dense_52/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
%sequential_13/dense_52/Tensordot/ProdProd2sequential_13/dense_52/Tensordot/GatherV2:output:0/sequential_13/dense_52/Tensordot/Const:output:0*
T0*
_output_shapes
: r
(sequential_13/dense_52/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
'sequential_13/dense_52/Tensordot/Prod_1Prod4sequential_13/dense_52/Tensordot/GatherV2_1:output:01sequential_13/dense_52/Tensordot/Const_1:output:0*
T0*
_output_shapes
: n
,sequential_13/dense_52/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
'sequential_13/dense_52/Tensordot/concatConcatV2.sequential_13/dense_52/Tensordot/free:output:0.sequential_13/dense_52/Tensordot/axes:output:05sequential_13/dense_52/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
&sequential_13/dense_52/Tensordot/stackPack.sequential_13/dense_52/Tensordot/Prod:output:00sequential_13/dense_52/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
*sequential_13/dense_52/Tensordot/transpose	Transposelayer_normalization_26/add:z:00sequential_13/dense_52/Tensordot/concat:output:0*
T0*-
_output_shapes
:????????????
(sequential_13/dense_52/Tensordot/ReshapeReshape.sequential_13/dense_52/Tensordot/transpose:y:0/sequential_13/dense_52/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
'sequential_13/dense_52/Tensordot/MatMulMatMul1sequential_13/dense_52/Tensordot/Reshape:output:07sequential_13/dense_52/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
(sequential_13/dense_52/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?p
.sequential_13/dense_52/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
)sequential_13/dense_52/Tensordot/concat_1ConcatV22sequential_13/dense_52/Tensordot/GatherV2:output:01sequential_13/dense_52/Tensordot/Const_2:output:07sequential_13/dense_52/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
 sequential_13/dense_52/TensordotReshape1sequential_13/dense_52/Tensordot/MatMul:product:02sequential_13/dense_52/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:????????????
-sequential_13/dense_52/BiasAdd/ReadVariableOpReadVariableOp6sequential_13_dense_52_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_13/dense_52/BiasAddBiasAdd)sequential_13/dense_52/Tensordot:output:05sequential_13/dense_52/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:????????????
sequential_13/dense_52/ReluRelu'sequential_13/dense_52/BiasAdd:output:0*
T0*-
_output_shapes
:????????????
/sequential_13/dense_53/Tensordot/ReadVariableOpReadVariableOp8sequential_13_dense_53_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype0o
%sequential_13/dense_53/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:v
%sequential_13/dense_53/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
&sequential_13/dense_53/Tensordot/ShapeShape)sequential_13/dense_52/Relu:activations:0*
T0*
_output_shapes
:p
.sequential_13/dense_53/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
)sequential_13/dense_53/Tensordot/GatherV2GatherV2/sequential_13/dense_53/Tensordot/Shape:output:0.sequential_13/dense_53/Tensordot/free:output:07sequential_13/dense_53/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:r
0sequential_13/dense_53/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
+sequential_13/dense_53/Tensordot/GatherV2_1GatherV2/sequential_13/dense_53/Tensordot/Shape:output:0.sequential_13/dense_53/Tensordot/axes:output:09sequential_13/dense_53/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
&sequential_13/dense_53/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
%sequential_13/dense_53/Tensordot/ProdProd2sequential_13/dense_53/Tensordot/GatherV2:output:0/sequential_13/dense_53/Tensordot/Const:output:0*
T0*
_output_shapes
: r
(sequential_13/dense_53/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
'sequential_13/dense_53/Tensordot/Prod_1Prod4sequential_13/dense_53/Tensordot/GatherV2_1:output:01sequential_13/dense_53/Tensordot/Const_1:output:0*
T0*
_output_shapes
: n
,sequential_13/dense_53/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
'sequential_13/dense_53/Tensordot/concatConcatV2.sequential_13/dense_53/Tensordot/free:output:0.sequential_13/dense_53/Tensordot/axes:output:05sequential_13/dense_53/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
&sequential_13/dense_53/Tensordot/stackPack.sequential_13/dense_53/Tensordot/Prod:output:00sequential_13/dense_53/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
*sequential_13/dense_53/Tensordot/transpose	Transpose)sequential_13/dense_52/Relu:activations:00sequential_13/dense_53/Tensordot/concat:output:0*
T0*-
_output_shapes
:????????????
(sequential_13/dense_53/Tensordot/ReshapeReshape.sequential_13/dense_53/Tensordot/transpose:y:0/sequential_13/dense_53/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
'sequential_13/dense_53/Tensordot/MatMulMatMul1sequential_13/dense_53/Tensordot/Reshape:output:07sequential_13/dense_53/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
(sequential_13/dense_53/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?p
.sequential_13/dense_53/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
)sequential_13/dense_53/Tensordot/concat_1ConcatV22sequential_13/dense_53/Tensordot/GatherV2:output:01sequential_13/dense_53/Tensordot/Const_2:output:07sequential_13/dense_53/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
 sequential_13/dense_53/TensordotReshape1sequential_13/dense_53/Tensordot/MatMul:product:02sequential_13/dense_53/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:????????????
-sequential_13/dense_53/BiasAdd/ReadVariableOpReadVariableOp6sequential_13_dense_53_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_13/dense_53/BiasAddBiasAdd)sequential_13/dense_53/Tensordot:output:05sequential_13/dense_53/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????]
dropout_56/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout_56/dropout/MulMul'sequential_13/dense_53/BiasAdd:output:0!dropout_56/dropout/Const:output:0*
T0*-
_output_shapes
:???????????o
dropout_56/dropout/ShapeShape'sequential_13/dense_53/BiasAdd:output:0*
T0*
_output_shapes
:?
/dropout_56/dropout/random_uniform/RandomUniformRandomUniform!dropout_56/dropout/Shape:output:0*
T0*-
_output_shapes
:???????????*
dtype0f
!dropout_56/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout_56/dropout/GreaterEqualGreaterEqual8dropout_56/dropout/random_uniform/RandomUniform:output:0*dropout_56/dropout/GreaterEqual/y:output:0*
T0*-
_output_shapes
:????????????
dropout_56/dropout/CastCast#dropout_56/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*-
_output_shapes
:????????????
dropout_56/dropout/Mul_1Muldropout_56/dropout/Mul:z:0dropout_56/dropout/Cast:y:0*
T0*-
_output_shapes
:????????????
add_1AddV2layer_normalization_26/add:z:0dropout_56/dropout/Mul_1:z:0*
T0*-
_output_shapes
:???????????U
layer_normalization_27/ShapeShape	add_1:z:0*
T0*
_output_shapes
:t
*layer_normalization_27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,layer_normalization_27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,layer_normalization_27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$layer_normalization_27/strided_sliceStridedSlice%layer_normalization_27/Shape:output:03layer_normalization_27/strided_slice/stack:output:05layer_normalization_27/strided_slice/stack_1:output:05layer_normalization_27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
layer_normalization_27/mul/xConst*
_output_shapes
: *
dtype0*
value	B :?
layer_normalization_27/mulMul%layer_normalization_27/mul/x:output:0-layer_normalization_27/strided_slice:output:0*
T0*
_output_shapes
: v
,layer_normalization_27/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_27/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_27/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_27/strided_slice_1StridedSlice%layer_normalization_27/Shape:output:05layer_normalization_27/strided_slice_1/stack:output:07layer_normalization_27/strided_slice_1/stack_1:output:07layer_normalization_27/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
layer_normalization_27/mul_1Mullayer_normalization_27/mul:z:0/layer_normalization_27/strided_slice_1:output:0*
T0*
_output_shapes
: v
,layer_normalization_27/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_27/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.layer_normalization_27/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_27/strided_slice_2StridedSlice%layer_normalization_27/Shape:output:05layer_normalization_27/strided_slice_2/stack:output:07layer_normalization_27/strided_slice_2/stack_1:output:07layer_normalization_27/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
layer_normalization_27/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :?
layer_normalization_27/mul_2Mul'layer_normalization_27/mul_2/x:output:0/layer_normalization_27/strided_slice_2:output:0*
T0*
_output_shapes
: h
&layer_normalization_27/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :h
&layer_normalization_27/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
$layer_normalization_27/Reshape/shapePack/layer_normalization_27/Reshape/shape/0:output:0 layer_normalization_27/mul_1:z:0 layer_normalization_27/mul_2:z:0/layer_normalization_27/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
layer_normalization_27/ReshapeReshape	add_1:z:0-layer_normalization_27/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????e
"layer_normalization_27/ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :??
 layer_normalization_27/ones/LessLess layer_normalization_27/mul_1:z:0+layer_normalization_27/ones/Less/y:output:0*
T0*
_output_shapes
: z
"layer_normalization_27/ones/packedPack layer_normalization_27/mul_1:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_27/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
layer_normalization_27/onesFill+layer_normalization_27/ones/packed:output:0*layer_normalization_27/ones/Const:output:0*
T0*#
_output_shapes
:?????????f
#layer_normalization_27/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :??
!layer_normalization_27/zeros/LessLess layer_normalization_27/mul_1:z:0,layer_normalization_27/zeros/Less/y:output:0*
T0*
_output_shapes
: {
#layer_normalization_27/zeros/packedPack layer_normalization_27/mul_1:z:0*
N*
T0*
_output_shapes
:g
"layer_normalization_27/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
layer_normalization_27/zerosFill,layer_normalization_27/zeros/packed:output:0+layer_normalization_27/zeros/Const:output:0*
T0*#
_output_shapes
:?????????_
layer_normalization_27/ConstConst*
_output_shapes
: *
dtype0*
valueB a
layer_normalization_27/Const_1Const*
_output_shapes
: *
dtype0*
valueB ?
'layer_normalization_27/FusedBatchNormV3FusedBatchNormV3'layer_normalization_27/Reshape:output:0$layer_normalization_27/ones:output:0%layer_normalization_27/zeros:output:0%layer_normalization_27/Const:output:0'layer_normalization_27/Const_1:output:0*
T0*
U0*p
_output_shapes^
\:??????????:?????????:?????????:?????????:?????????:*
data_formatNCHW?
 layer_normalization_27/Reshape_1Reshape+layer_normalization_27/FusedBatchNormV3:y:0%layer_normalization_27/Shape:output:0*
T0*-
_output_shapes
:????????????
+layer_normalization_27/mul_3/ReadVariableOpReadVariableOp4layer_normalization_27_mul_3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
layer_normalization_27/mul_3Mul)layer_normalization_27/Reshape_1:output:03layer_normalization_27/mul_3/ReadVariableOp:value:0*
T0*-
_output_shapes
:????????????
)layer_normalization_27/add/ReadVariableOpReadVariableOp2layer_normalization_27_add_readvariableop_resource*
_output_shapes	
:?*
dtype0?
layer_normalization_27/addAddV2 layer_normalization_27/mul_3:z:01layer_normalization_27/add/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????s
IdentityIdentitylayer_normalization_27/add:z:0^NoOp*
T0*-
_output_shapes
:????????????
NoOpNoOp*^layer_normalization_26/add/ReadVariableOp,^layer_normalization_26/mul_3/ReadVariableOp*^layer_normalization_27/add/ReadVariableOp,^layer_normalization_27/mul_3/ReadVariableOp<^multi_head_attention_13/attention_output/add/ReadVariableOpF^multi_head_attention_13/attention_output/einsum/Einsum/ReadVariableOp/^multi_head_attention_13/key/add/ReadVariableOp9^multi_head_attention_13/key/einsum/Einsum/ReadVariableOp1^multi_head_attention_13/query/add/ReadVariableOp;^multi_head_attention_13/query/einsum/Einsum/ReadVariableOp1^multi_head_attention_13/value/add/ReadVariableOp;^multi_head_attention_13/value/einsum/Einsum/ReadVariableOp.^sequential_13/dense_52/BiasAdd/ReadVariableOp0^sequential_13/dense_52/Tensordot/ReadVariableOp.^sequential_13/dense_53/BiasAdd/ReadVariableOp0^sequential_13/dense_53/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:???????????: : : : : : : : : : : : : : : : 2V
)layer_normalization_26/add/ReadVariableOp)layer_normalization_26/add/ReadVariableOp2Z
+layer_normalization_26/mul_3/ReadVariableOp+layer_normalization_26/mul_3/ReadVariableOp2V
)layer_normalization_27/add/ReadVariableOp)layer_normalization_27/add/ReadVariableOp2Z
+layer_normalization_27/mul_3/ReadVariableOp+layer_normalization_27/mul_3/ReadVariableOp2z
;multi_head_attention_13/attention_output/add/ReadVariableOp;multi_head_attention_13/attention_output/add/ReadVariableOp2?
Emulti_head_attention_13/attention_output/einsum/Einsum/ReadVariableOpEmulti_head_attention_13/attention_output/einsum/Einsum/ReadVariableOp2`
.multi_head_attention_13/key/add/ReadVariableOp.multi_head_attention_13/key/add/ReadVariableOp2t
8multi_head_attention_13/key/einsum/Einsum/ReadVariableOp8multi_head_attention_13/key/einsum/Einsum/ReadVariableOp2d
0multi_head_attention_13/query/add/ReadVariableOp0multi_head_attention_13/query/add/ReadVariableOp2x
:multi_head_attention_13/query/einsum/Einsum/ReadVariableOp:multi_head_attention_13/query/einsum/Einsum/ReadVariableOp2d
0multi_head_attention_13/value/add/ReadVariableOp0multi_head_attention_13/value/add/ReadVariableOp2x
:multi_head_attention_13/value/einsum/Einsum/ReadVariableOp:multi_head_attention_13/value/einsum/Einsum/ReadVariableOp2^
-sequential_13/dense_52/BiasAdd/ReadVariableOp-sequential_13/dense_52/BiasAdd/ReadVariableOp2b
/sequential_13/dense_52/Tensordot/ReadVariableOp/sequential_13/dense_52/Tensordot/ReadVariableOp2^
-sequential_13/dense_53/BiasAdd/ReadVariableOp-sequential_13/dense_53/BiasAdd/ReadVariableOp2b
/sequential_13/dense_53/Tensordot/ReadVariableOp/sequential_13/dense_53/Tensordot/ReadVariableOp:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_58_layer_call_and_return_conditional_losses_83586

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
H__inference_sequential_13_layer_call_and_return_conditional_losses_81234

inputs"
dense_52_81192:
??
dense_52_81194:	?"
dense_53_81228:
??
dense_53_81230:	?
identity?? dense_52/StatefulPartitionedCall? dense_53/StatefulPartitionedCall?
 dense_52/StatefulPartitionedCallStatefulPartitionedCallinputsdense_52_81192dense_52_81194*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_52_layer_call_and_return_conditional_losses_81191?
 dense_53/StatefulPartitionedCallStatefulPartitionedCall)dense_52/StatefulPartitionedCall:output:0dense_53_81228dense_53_81230*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_53_layer_call_and_return_conditional_losses_81227~
IdentityIdentity)dense_53/StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:????????????
NoOpNoOp!^dense_52/StatefulPartitionedCall!^dense_53/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : : : 2D
 dense_52/StatefulPartitionedCall dense_52/StatefulPartitionedCall2D
 dense_53/StatefulPartitionedCall dense_53/StatefulPartitionedCall:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
-__inference_sequential_13_layer_call_fn_81318
dense_52_input
unknown:
??
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_52_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_13_layer_call_and_return_conditional_losses_81294u
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
-
_output_shapes
:???????????
(
_user_specified_namedense_52_input
?
c
*__inference_dropout_57_layer_call_fn_83534

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_57_layer_call_and_return_conditional_losses_81781p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
>
input_142
serving_default_input_14:0??????????<
dense_550
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api

signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
?
	token_emb
pos_emb
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
att
ffn

layernorm1

layernorm2
dropout1
dropout2
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	variables
 trainable_variables
!regularization_losses
"	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
#	variables
$trainable_variables
%regularization_losses
&	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
-	variables
.trainable_variables
/regularization_losses
0	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

1kernel
2bias
3	variables
4trainable_variables
5regularization_losses
6	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
7iter

8beta_1

9beta_2
	:decay
;learning_rate'm?(m?1m?2m?<m?=m?>m??m?@m?Am?Bm?Cm?Dm?Em?Fm?Gm?Hm?Im?Jm?Km?Lm?Mm?'v?(v?1v?2v?<v?=v?>v??v?@v?Av?Bv?Cv?Dv?Ev?Fv?Gv?Hv?Iv?Jv?Kv?Lv?Mv?"
	optimizer
?
<0
=1
>2
?3
@4
A5
B6
C7
D8
E9
F10
G11
H12
I13
J14
K15
L16
M17
'18
(19
120
221"
trackable_list_wrapper
?
<0
=1
>2
?3
@4
A5
B6
C7
D8
E9
F10
G11
H12
I13
J14
K15
L16
M17
'18
(19
120
221"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics

	variables
trainable_variables
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?
<
embeddings
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
=
embeddings
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
`_query_dense
a
_key_dense
b_value_dense
c_softmax
d_dropout_layer
e_output_dense
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
jlayer_with_weights-0
jlayer-0
klayer_with_weights-1
klayer-1
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_sequential
?
paxis
	Jgamma
Kbeta
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
uaxis
	Lgamma
Mbeta
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
z	variables
{trainable_variables
|regularization_losses
}	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
~	variables
trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
>0
?1
@2
A3
B4
C5
D6
E7
F8
G9
H10
I11
J12
K13
L14
M15"
trackable_list_wrapper
?
>0
?1
@2
A3
B4
C5
D6
E7
F8
G9
H10
I11
J12
K13
L14
M15"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
 trainable_variables
!regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
#	variables
$trainable_variables
%regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?2dense_54/kernel
:2dense_54/bias
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
)	variables
*trainable_variables
+regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
-	variables
.trainable_variables
/regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:2dense_55/kernel
:2dense_55/bias
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
3	variables
4trainable_variables
5regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
K:I
?)?27token_and_position_embedding_13/embedding_26/embeddings
K:I
??27token_and_position_embedding_13/embedding_27/embeddings
Q:O??29transformer_block_13/multi_head_attention_13/query/kernel
J:H	?27transformer_block_13/multi_head_attention_13/query/bias
O:M??27transformer_block_13/multi_head_attention_13/key/kernel
H:F	?25transformer_block_13/multi_head_attention_13/key/bias
Q:O??29transformer_block_13/multi_head_attention_13/value/kernel
J:H	?27transformer_block_13/multi_head_attention_13/value/bias
\:Z??2Dtransformer_block_13/multi_head_attention_13/attention_output/kernel
Q:O?2Btransformer_block_13/multi_head_attention_13/attention_output/bias
#:!
??2dense_52/kernel
:?2dense_52/bias
#:!
??2dense_53/kernel
:?2dense_53/bias
@:>?21transformer_block_13/layer_normalization_26/gamma
?:=?20transformer_block_13/layer_normalization_26/beta
@:>?21transformer_block_13/layer_normalization_27/gamma
?:=?20transformer_block_13/layer_normalization_27/beta
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
<0"
trackable_list_wrapper
'
<0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
=0"
trackable_list_wrapper
'
=0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?
?partial_output_shape
?full_output_shape

>kernel
?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?partial_output_shape
?full_output_shape

@kernel
Abias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?partial_output_shape
?full_output_shape

Bkernel
Cbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?partial_output_shape
?full_output_shape

Dkernel
Ebias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
X
>0
?1
@2
A3
B4
C5
D6
E7"
trackable_list_wrapper
X
>0
?1
@2
A3
B4
C5
D6
E7"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
f	variables
gtrainable_variables
hregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

Fkernel
Gbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Hkernel
Ibias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
<
F0
G1
H2
I3"
trackable_list_wrapper
<
F0
G1
H2
I3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
l	variables
mtrainable_variables
nregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
q	variables
rtrainable_variables
sregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
v	variables
wtrainable_variables
xregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
z	variables
{trainable_variables
|regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
~	variables
trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
J
`0
a1
b2
c3
d4
e5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
H0
I1"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
':%	?2Adam/dense_54/kernel/m
 :2Adam/dense_54/bias/m
&:$2Adam/dense_55/kernel/m
 :2Adam/dense_55/bias/m
P:N
?)?2>Adam/token_and_position_embedding_13/embedding_26/embeddings/m
P:N
??2>Adam/token_and_position_embedding_13/embedding_27/embeddings/m
V:T??2@Adam/transformer_block_13/multi_head_attention_13/query/kernel/m
O:M	?2>Adam/transformer_block_13/multi_head_attention_13/query/bias/m
T:R??2>Adam/transformer_block_13/multi_head_attention_13/key/kernel/m
M:K	?2<Adam/transformer_block_13/multi_head_attention_13/key/bias/m
V:T??2@Adam/transformer_block_13/multi_head_attention_13/value/kernel/m
O:M	?2>Adam/transformer_block_13/multi_head_attention_13/value/bias/m
a:_??2KAdam/transformer_block_13/multi_head_attention_13/attention_output/kernel/m
V:T?2IAdam/transformer_block_13/multi_head_attention_13/attention_output/bias/m
(:&
??2Adam/dense_52/kernel/m
!:?2Adam/dense_52/bias/m
(:&
??2Adam/dense_53/kernel/m
!:?2Adam/dense_53/bias/m
E:C?28Adam/transformer_block_13/layer_normalization_26/gamma/m
D:B?27Adam/transformer_block_13/layer_normalization_26/beta/m
E:C?28Adam/transformer_block_13/layer_normalization_27/gamma/m
D:B?27Adam/transformer_block_13/layer_normalization_27/beta/m
':%	?2Adam/dense_54/kernel/v
 :2Adam/dense_54/bias/v
&:$2Adam/dense_55/kernel/v
 :2Adam/dense_55/bias/v
P:N
?)?2>Adam/token_and_position_embedding_13/embedding_26/embeddings/v
P:N
??2>Adam/token_and_position_embedding_13/embedding_27/embeddings/v
V:T??2@Adam/transformer_block_13/multi_head_attention_13/query/kernel/v
O:M	?2>Adam/transformer_block_13/multi_head_attention_13/query/bias/v
T:R??2>Adam/transformer_block_13/multi_head_attention_13/key/kernel/v
M:K	?2<Adam/transformer_block_13/multi_head_attention_13/key/bias/v
V:T??2@Adam/transformer_block_13/multi_head_attention_13/value/kernel/v
O:M	?2>Adam/transformer_block_13/multi_head_attention_13/value/bias/v
a:_??2KAdam/transformer_block_13/multi_head_attention_13/attention_output/kernel/v
V:T?2IAdam/transformer_block_13/multi_head_attention_13/attention_output/bias/v
(:&
??2Adam/dense_52/kernel/v
!:?2Adam/dense_52/bias/v
(:&
??2Adam/dense_53/kernel/v
!:?2Adam/dense_53/bias/v
E:C?28Adam/transformer_block_13/layer_normalization_26/gamma/v
D:B?27Adam/transformer_block_13/layer_normalization_26/beta/v
E:C?28Adam/transformer_block_13/layer_normalization_27/gamma/v
D:B?27Adam/transformer_block_13/layer_normalization_27/beta/v
?2?
(__inference_model_13_layer_call_fn_81718
(__inference_model_13_layer_call_fn_82490
(__inference_model_13_layer_call_fn_82539
(__inference_model_13_layer_call_fn_82274?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_model_13_layer_call_and_return_conditional_losses_82762
C__inference_model_13_layer_call_and_return_conditional_losses_83012
C__inference_model_13_layer_call_and_return_conditional_losses_82329
C__inference_model_13_layer_call_and_return_conditional_losses_82384?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
 __inference__wrapped_model_81153input_14"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
?__inference_token_and_position_embedding_13_layer_call_fn_83021?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
Z__inference_token_and_position_embedding_13_layer_call_and_return_conditional_losses_83045?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_transformer_block_13_layer_call_fn_83082
4__inference_transformer_block_13_layer_call_fn_83119?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults? 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
O__inference_transformer_block_13_layer_call_and_return_conditional_losses_83304
O__inference_transformer_block_13_layer_call_and_return_conditional_losses_83502?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults? 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
;__inference_global_average_pooling1d_13_layer_call_fn_83507
;__inference_global_average_pooling1d_13_layer_call_fn_83512?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
V__inference_global_average_pooling1d_13_layer_call_and_return_conditional_losses_83518
V__inference_global_average_pooling1d_13_layer_call_and_return_conditional_losses_83524?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dropout_57_layer_call_fn_83529
*__inference_dropout_57_layer_call_fn_83534?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dropout_57_layer_call_and_return_conditional_losses_83539
E__inference_dropout_57_layer_call_and_return_conditional_losses_83551?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_dense_54_layer_call_fn_83560?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_54_layer_call_and_return_conditional_losses_83571?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dropout_58_layer_call_fn_83576
*__inference_dropout_58_layer_call_fn_83581?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dropout_58_layer_call_and_return_conditional_losses_83586
E__inference_dropout_58_layer_call_and_return_conditional_losses_83598?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_dense_55_layer_call_fn_83607?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_55_layer_call_and_return_conditional_losses_83618?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_82441input_14"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpece
args]?Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults?

 

 
p 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpece
args]?Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults?

 

 
p 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
-__inference_sequential_13_layer_call_fn_81245
-__inference_sequential_13_layer_call_fn_83631
-__inference_sequential_13_layer_call_fn_83644
-__inference_sequential_13_layer_call_fn_81318?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_sequential_13_layer_call_and_return_conditional_losses_83701
H__inference_sequential_13_layer_call_and_return_conditional_losses_83758
H__inference_sequential_13_layer_call_and_return_conditional_losses_81332
H__inference_sequential_13_layer_call_and_return_conditional_losses_81346?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_52_layer_call_fn_83767?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_52_layer_call_and_return_conditional_losses_83798?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_53_layer_call_fn_83807?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_53_layer_call_and_return_conditional_losses_83837?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_81153?=<>?@ABCDEJKFGHILM'(122?/
(?%
#? 
input_14??????????
? "3?0
.
dense_55"?
dense_55??????????
C__inference_dense_52_layer_call_and_return_conditional_losses_83798hFG5?2
+?(
&?#
inputs???????????
? "+?(
!?
0???????????
? ?
(__inference_dense_52_layer_call_fn_83767[FG5?2
+?(
&?#
inputs???????????
? "?????????????
C__inference_dense_53_layer_call_and_return_conditional_losses_83837hHI5?2
+?(
&?#
inputs???????????
? "+?(
!?
0???????????
? ?
(__inference_dense_53_layer_call_fn_83807[HI5?2
+?(
&?#
inputs???????????
? "?????????????
C__inference_dense_54_layer_call_and_return_conditional_losses_83571]'(0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? |
(__inference_dense_54_layer_call_fn_83560P'(0?-
&?#
!?
inputs??????????
? "???????????
C__inference_dense_55_layer_call_and_return_conditional_losses_83618\12/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? {
(__inference_dense_55_layer_call_fn_83607O12/?,
%?"
 ?
inputs?????????
? "???????????
E__inference_dropout_57_layer_call_and_return_conditional_losses_83539^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
E__inference_dropout_57_layer_call_and_return_conditional_losses_83551^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? 
*__inference_dropout_57_layer_call_fn_83529Q4?1
*?'
!?
inputs??????????
p 
? "???????????
*__inference_dropout_57_layer_call_fn_83534Q4?1
*?'
!?
inputs??????????
p
? "????????????
E__inference_dropout_58_layer_call_and_return_conditional_losses_83586\3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
E__inference_dropout_58_layer_call_and_return_conditional_losses_83598\3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? }
*__inference_dropout_58_layer_call_fn_83576O3?0
)?&
 ?
inputs?????????
p 
? "??????????}
*__inference_dropout_58_layer_call_fn_83581O3?0
)?&
 ?
inputs?????????
p
? "???????????
V__inference_global_average_pooling1d_13_layer_call_and_return_conditional_losses_83518{I?F
??<
6?3
inputs'???????????????????????????

 
? ".?+
$?!
0??????????????????
? ?
V__inference_global_average_pooling1d_13_layer_call_and_return_conditional_losses_83524c9?6
/?,
&?#
inputs???????????

 
? "&?#
?
0??????????
? ?
;__inference_global_average_pooling1d_13_layer_call_fn_83507nI?F
??<
6?3
inputs'???????????????????????????

 
? "!????????????????????
;__inference_global_average_pooling1d_13_layer_call_fn_83512V9?6
/?,
&?#
inputs???????????

 
? "????????????
C__inference_model_13_layer_call_and_return_conditional_losses_82329{=<>?@ABCDEJKFGHILM'(12:?7
0?-
#? 
input_14??????????
p 

 
? "%?"
?
0?????????
? ?
C__inference_model_13_layer_call_and_return_conditional_losses_82384{=<>?@ABCDEJKFGHILM'(12:?7
0?-
#? 
input_14??????????
p

 
? "%?"
?
0?????????
? ?
C__inference_model_13_layer_call_and_return_conditional_losses_82762y=<>?@ABCDEJKFGHILM'(128?5
.?+
!?
inputs??????????
p 

 
? "%?"
?
0?????????
? ?
C__inference_model_13_layer_call_and_return_conditional_losses_83012y=<>?@ABCDEJKFGHILM'(128?5
.?+
!?
inputs??????????
p

 
? "%?"
?
0?????????
? ?
(__inference_model_13_layer_call_fn_81718n=<>?@ABCDEJKFGHILM'(12:?7
0?-
#? 
input_14??????????
p 

 
? "???????????
(__inference_model_13_layer_call_fn_82274n=<>?@ABCDEJKFGHILM'(12:?7
0?-
#? 
input_14??????????
p

 
? "???????????
(__inference_model_13_layer_call_fn_82490l=<>?@ABCDEJKFGHILM'(128?5
.?+
!?
inputs??????????
p 

 
? "???????????
(__inference_model_13_layer_call_fn_82539l=<>?@ABCDEJKFGHILM'(128?5
.?+
!?
inputs??????????
p

 
? "???????????
H__inference_sequential_13_layer_call_and_return_conditional_losses_81332zFGHIE?B
;?8
.?+
dense_52_input???????????
p 

 
? "+?(
!?
0???????????
? ?
H__inference_sequential_13_layer_call_and_return_conditional_losses_81346zFGHIE?B
;?8
.?+
dense_52_input???????????
p

 
? "+?(
!?
0???????????
? ?
H__inference_sequential_13_layer_call_and_return_conditional_losses_83701rFGHI=?:
3?0
&?#
inputs???????????
p 

 
? "+?(
!?
0???????????
? ?
H__inference_sequential_13_layer_call_and_return_conditional_losses_83758rFGHI=?:
3?0
&?#
inputs???????????
p

 
? "+?(
!?
0???????????
? ?
-__inference_sequential_13_layer_call_fn_81245mFGHIE?B
;?8
.?+
dense_52_input???????????
p 

 
? "?????????????
-__inference_sequential_13_layer_call_fn_81318mFGHIE?B
;?8
.?+
dense_52_input???????????
p

 
? "?????????????
-__inference_sequential_13_layer_call_fn_83631eFGHI=?:
3?0
&?#
inputs???????????
p 

 
? "?????????????
-__inference_sequential_13_layer_call_fn_83644eFGHI=?:
3?0
&?#
inputs???????????
p

 
? "?????????????
#__inference_signature_wrapper_82441?=<>?@ABCDEJKFGHILM'(12>?;
? 
4?1
/
input_14#? 
input_14??????????"3?0
.
dense_55"?
dense_55??????????
Z__inference_token_and_position_embedding_13_layer_call_and_return_conditional_losses_83045^=<+?(
!?
?
x??????????
? "+?(
!?
0???????????
? ?
?__inference_token_and_position_embedding_13_layer_call_fn_83021Q=<+?(
!?
?
x??????????
? "?????????????
O__inference_transformer_block_13_layer_call_and_return_conditional_losses_83304z>?@ABCDEJKFGHILM9?6
/?,
&?#
inputs???????????
p 
? "+?(
!?
0???????????
? ?
O__inference_transformer_block_13_layer_call_and_return_conditional_losses_83502z>?@ABCDEJKFGHILM9?6
/?,
&?#
inputs???????????
p
? "+?(
!?
0???????????
? ?
4__inference_transformer_block_13_layer_call_fn_83082m>?@ABCDEJKFGHILM9?6
/?,
&?#
inputs???????????
p 
? "?????????????
4__inference_transformer_block_13_layer_call_fn_83119m>?@ABCDEJKFGHILM9?6
/?,
&?#
inputs???????????
p
? "????????????