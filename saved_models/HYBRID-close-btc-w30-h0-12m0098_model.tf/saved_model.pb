т;
Ќ
B
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
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
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
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
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	
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
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype

SplitV

value"T
size_splits"Tlen
	split_dim
output"T*	num_split"
	num_splitint(0"	
Ttype"
Tlentype0	:
2	
О
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
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
і
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
;
Sub
x"T
y"T
z"T"
Ttype:
2	

TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type
output_handle"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements

handle"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintџџџџџџџџџ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
"serve*2.4.12unknown8е§7
z
dense_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2 * 
shared_namedense_27/kernel
s
#dense_27/kernel/Read/ReadVariableOpReadVariableOpdense_27/kernel*
_output_shapes

:2 *
dtype0
r
dense_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_27/bias
k
!dense_27/bias/Read/ReadVariableOpReadVariableOpdense_27/bias*
_output_shapes
: *
dtype0
z
dense_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2@* 
shared_namedense_28/kernel
s
#dense_28/kernel/Read/ReadVariableOpReadVariableOpdense_28/kernel*
_output_shapes

:2@*
dtype0
r
dense_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_28/bias
k
!dense_28/bias/Read/ReadVariableOpReadVariableOpdense_28/bias*
_output_shapes
:@*
dtype0
z
dense_29/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`* 
shared_namedense_29/kernel
s
#dense_29/kernel/Read/ReadVariableOpReadVariableOpdense_29/kernel*
_output_shapes

:`*
dtype0
r
dense_29/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_29/bias
k
!dense_29/bias/Read/ReadVariableOpReadVariableOpdense_29/bias*
_output_shapes
:*
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

lstm_18/lstm_cell_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ќ*,
shared_namelstm_18/lstm_cell_18/kernel

/lstm_18/lstm_cell_18/kernel/Read/ReadVariableOpReadVariableOplstm_18/lstm_cell_18/kernel*
_output_shapes
:	Ќ*
dtype0
Ї
%lstm_18/lstm_cell_18/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	KЌ*6
shared_name'%lstm_18/lstm_cell_18/recurrent_kernel
 
9lstm_18/lstm_cell_18/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_18/lstm_cell_18/recurrent_kernel*
_output_shapes
:	KЌ*
dtype0

lstm_18/lstm_cell_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ќ**
shared_namelstm_18/lstm_cell_18/bias

-lstm_18/lstm_cell_18/bias/Read/ReadVariableOpReadVariableOplstm_18/lstm_cell_18/bias*
_output_shapes	
:Ќ*
dtype0

gru_9/gru_cell_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*(
shared_namegru_9/gru_cell_9/kernel

+gru_9/gru_cell_9/kernel/Read/ReadVariableOpReadVariableOpgru_9/gru_cell_9/kernel*
_output_shapes
:	*
dtype0

!gru_9/gru_cell_9/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2*2
shared_name#!gru_9/gru_cell_9/recurrent_kernel

5gru_9/gru_cell_9/recurrent_kernel/Read/ReadVariableOpReadVariableOp!gru_9/gru_cell_9/recurrent_kernel*
_output_shapes
:	2*
dtype0

gru_9/gru_cell_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*&
shared_namegru_9/gru_cell_9/bias

)gru_9/gru_cell_9/bias/Read/ReadVariableOpReadVariableOpgru_9/gru_cell_9/bias*
_output_shapes
:	*
dtype0

lstm_19/lstm_cell_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	KШ*,
shared_namelstm_19/lstm_cell_19/kernel

/lstm_19/lstm_cell_19/kernel/Read/ReadVariableOpReadVariableOplstm_19/lstm_cell_19/kernel*
_output_shapes
:	KШ*
dtype0
Ї
%lstm_19/lstm_cell_19/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Ш*6
shared_name'%lstm_19/lstm_cell_19/recurrent_kernel
 
9lstm_19/lstm_cell_19/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_19/lstm_cell_19/recurrent_kernel*
_output_shapes
:	2Ш*
dtype0

lstm_19/lstm_cell_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш**
shared_namelstm_19/lstm_cell_19/bias

-lstm_19/lstm_cell_19/bias/Read/ReadVariableOpReadVariableOplstm_19/lstm_cell_19/bias*
_output_shapes	
:Ш*
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
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0

Adam/dense_27/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2 *'
shared_nameAdam/dense_27/kernel/m

*Adam/dense_27/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_27/kernel/m*
_output_shapes

:2 *
dtype0

Adam/dense_27/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_27/bias/m
y
(Adam/dense_27/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_27/bias/m*
_output_shapes
: *
dtype0

Adam/dense_28/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2@*'
shared_nameAdam/dense_28/kernel/m

*Adam/dense_28/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_28/kernel/m*
_output_shapes

:2@*
dtype0

Adam/dense_28/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_28/bias/m
y
(Adam/dense_28/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_28/bias/m*
_output_shapes
:@*
dtype0

Adam/dense_29/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*'
shared_nameAdam/dense_29/kernel/m

*Adam/dense_29/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_29/kernel/m*
_output_shapes

:`*
dtype0

Adam/dense_29/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_29/bias/m
y
(Adam/dense_29/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_29/bias/m*
_output_shapes
:*
dtype0
Ё
"Adam/lstm_18/lstm_cell_18/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ќ*3
shared_name$"Adam/lstm_18/lstm_cell_18/kernel/m

6Adam/lstm_18/lstm_cell_18/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_18/lstm_cell_18/kernel/m*
_output_shapes
:	Ќ*
dtype0
Е
,Adam/lstm_18/lstm_cell_18/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	KЌ*=
shared_name.,Adam/lstm_18/lstm_cell_18/recurrent_kernel/m
Ў
@Adam/lstm_18/lstm_cell_18/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_18/lstm_cell_18/recurrent_kernel/m*
_output_shapes
:	KЌ*
dtype0

 Adam/lstm_18/lstm_cell_18/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ќ*1
shared_name" Adam/lstm_18/lstm_cell_18/bias/m

4Adam/lstm_18/lstm_cell_18/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_18/lstm_cell_18/bias/m*
_output_shapes	
:Ќ*
dtype0

Adam/gru_9/gru_cell_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*/
shared_name Adam/gru_9/gru_cell_9/kernel/m

2Adam/gru_9/gru_cell_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/gru_9/gru_cell_9/kernel/m*
_output_shapes
:	*
dtype0
­
(Adam/gru_9/gru_cell_9/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2*9
shared_name*(Adam/gru_9/gru_cell_9/recurrent_kernel/m
І
<Adam/gru_9/gru_cell_9/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp(Adam/gru_9/gru_cell_9/recurrent_kernel/m*
_output_shapes
:	2*
dtype0

Adam/gru_9/gru_cell_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*-
shared_nameAdam/gru_9/gru_cell_9/bias/m

0Adam/gru_9/gru_cell_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/gru_9/gru_cell_9/bias/m*
_output_shapes
:	*
dtype0
Ё
"Adam/lstm_19/lstm_cell_19/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	KШ*3
shared_name$"Adam/lstm_19/lstm_cell_19/kernel/m

6Adam/lstm_19/lstm_cell_19/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_19/lstm_cell_19/kernel/m*
_output_shapes
:	KШ*
dtype0
Е
,Adam/lstm_19/lstm_cell_19/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Ш*=
shared_name.,Adam/lstm_19/lstm_cell_19/recurrent_kernel/m
Ў
@Adam/lstm_19/lstm_cell_19/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_19/lstm_cell_19/recurrent_kernel/m*
_output_shapes
:	2Ш*
dtype0

 Adam/lstm_19/lstm_cell_19/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*1
shared_name" Adam/lstm_19/lstm_cell_19/bias/m

4Adam/lstm_19/lstm_cell_19/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_19/lstm_cell_19/bias/m*
_output_shapes	
:Ш*
dtype0

Adam/dense_27/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2 *'
shared_nameAdam/dense_27/kernel/v

*Adam/dense_27/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_27/kernel/v*
_output_shapes

:2 *
dtype0

Adam/dense_27/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_27/bias/v
y
(Adam/dense_27/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_27/bias/v*
_output_shapes
: *
dtype0

Adam/dense_28/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2@*'
shared_nameAdam/dense_28/kernel/v

*Adam/dense_28/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_28/kernel/v*
_output_shapes

:2@*
dtype0

Adam/dense_28/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_28/bias/v
y
(Adam/dense_28/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_28/bias/v*
_output_shapes
:@*
dtype0

Adam/dense_29/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*'
shared_nameAdam/dense_29/kernel/v

*Adam/dense_29/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_29/kernel/v*
_output_shapes

:`*
dtype0

Adam/dense_29/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_29/bias/v
y
(Adam/dense_29/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_29/bias/v*
_output_shapes
:*
dtype0
Ё
"Adam/lstm_18/lstm_cell_18/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ќ*3
shared_name$"Adam/lstm_18/lstm_cell_18/kernel/v

6Adam/lstm_18/lstm_cell_18/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_18/lstm_cell_18/kernel/v*
_output_shapes
:	Ќ*
dtype0
Е
,Adam/lstm_18/lstm_cell_18/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	KЌ*=
shared_name.,Adam/lstm_18/lstm_cell_18/recurrent_kernel/v
Ў
@Adam/lstm_18/lstm_cell_18/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_18/lstm_cell_18/recurrent_kernel/v*
_output_shapes
:	KЌ*
dtype0

 Adam/lstm_18/lstm_cell_18/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ќ*1
shared_name" Adam/lstm_18/lstm_cell_18/bias/v

4Adam/lstm_18/lstm_cell_18/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_18/lstm_cell_18/bias/v*
_output_shapes	
:Ќ*
dtype0

Adam/gru_9/gru_cell_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*/
shared_name Adam/gru_9/gru_cell_9/kernel/v

2Adam/gru_9/gru_cell_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/gru_9/gru_cell_9/kernel/v*
_output_shapes
:	*
dtype0
­
(Adam/gru_9/gru_cell_9/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2*9
shared_name*(Adam/gru_9/gru_cell_9/recurrent_kernel/v
І
<Adam/gru_9/gru_cell_9/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp(Adam/gru_9/gru_cell_9/recurrent_kernel/v*
_output_shapes
:	2*
dtype0

Adam/gru_9/gru_cell_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*-
shared_nameAdam/gru_9/gru_cell_9/bias/v

0Adam/gru_9/gru_cell_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/gru_9/gru_cell_9/bias/v*
_output_shapes
:	*
dtype0
Ё
"Adam/lstm_19/lstm_cell_19/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	KШ*3
shared_name$"Adam/lstm_19/lstm_cell_19/kernel/v

6Adam/lstm_19/lstm_cell_19/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_19/lstm_cell_19/kernel/v*
_output_shapes
:	KШ*
dtype0
Е
,Adam/lstm_19/lstm_cell_19/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Ш*=
shared_name.,Adam/lstm_19/lstm_cell_19/recurrent_kernel/v
Ў
@Adam/lstm_19/lstm_cell_19/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_19/lstm_cell_19/recurrent_kernel/v*
_output_shapes
:	2Ш*
dtype0

 Adam/lstm_19/lstm_cell_19/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*1
shared_name" Adam/lstm_19/lstm_cell_19/bias/v

4Adam/lstm_19/lstm_cell_19/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_19/lstm_cell_19/bias/v*
_output_shapes	
:Ш*
dtype0

NoOpNoOp
џ]
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*К]
valueА]B­] BІ]

layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
l
cell

state_spec
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
l
cell

state_spec
trainable_variables
regularization_losses
	variables
 	keras_api
l
!cell
"
state_spec
#trainable_variables
$regularization_losses
%	variables
&	keras_api
R
'trainable_variables
(regularization_losses
)	variables
*	keras_api
h

+kernel
,bias
-trainable_variables
.regularization_losses
/	variables
0	keras_api
h

1kernel
2bias
3trainable_variables
4regularization_losses
5	variables
6	keras_api
R
7trainable_variables
8regularization_losses
9	variables
:	keras_api
h

;kernel
<bias
=trainable_variables
>regularization_losses
?	variables
@	keras_api
ь
Aiter

Bbeta_1

Cbeta_2
	Ddecay
Elearning_rate+mА,mБ1mВ2mГ;mД<mЕFmЖGmЗHmИImЙJmКKmЛLmМMmНNmО+vП,vР1vС2vТ;vУ<vФFvХGvЦHvЧIvШJvЩKvЪLvЫMvЬNvЭ
n
F0
G1
H2
I3
J4
K5
L6
M7
N8
+9
,10
111
212
;13
<14
 
n
F0
G1
H2
I3
J4
K5
L6
M7
N8
+9
,10
111
212
;13
<14
­
Olayer_metrics

Players
Qlayer_regularization_losses
trainable_variables
regularization_losses
Rnon_trainable_variables
Smetrics
	variables
 
~

Fkernel
Grecurrent_kernel
Hbias
Ttrainable_variables
Uregularization_losses
V	variables
W	keras_api
 

F0
G1
H2
 

F0
G1
H2
Й
Xlayer_metrics

Ylayers
Zlayer_regularization_losses
trainable_variables
regularization_losses
[non_trainable_variables
\metrics

]states
	variables
 
 
 
­
^layer_metrics

_layers
`layer_regularization_losses
trainable_variables
regularization_losses
anon_trainable_variables
bmetrics
	variables
~

Ikernel
Jrecurrent_kernel
Kbias
ctrainable_variables
dregularization_losses
e	variables
f	keras_api
 

I0
J1
K2
 

I0
J1
K2
Й
glayer_metrics

hlayers
ilayer_regularization_losses
trainable_variables
regularization_losses
jnon_trainable_variables
kmetrics

lstates
	variables
~

Lkernel
Mrecurrent_kernel
Nbias
mtrainable_variables
nregularization_losses
o	variables
p	keras_api
 

L0
M1
N2
 

L0
M1
N2
Й
qlayer_metrics

rlayers
slayer_regularization_losses
#trainable_variables
$regularization_losses
tnon_trainable_variables
umetrics

vstates
%	variables
 
 
 
­
wlayer_metrics

xlayers
ylayer_regularization_losses
'trainable_variables
(regularization_losses
znon_trainable_variables
{metrics
)	variables
[Y
VARIABLE_VALUEdense_27/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_27/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

+0
,1
 

+0
,1
Ў
|layer_metrics

}layers
~layer_regularization_losses
-trainable_variables
.regularization_losses
non_trainable_variables
metrics
/	variables
[Y
VARIABLE_VALUEdense_28/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_28/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

10
21
 

10
21
В
layer_metrics
layers
 layer_regularization_losses
3trainable_variables
4regularization_losses
non_trainable_variables
metrics
5	variables
 
 
 
В
layer_metrics
layers
 layer_regularization_losses
7trainable_variables
8regularization_losses
non_trainable_variables
metrics
9	variables
[Y
VARIABLE_VALUEdense_29/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_29/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

;0
<1
 

;0
<1
В
layer_metrics
layers
 layer_regularization_losses
=trainable_variables
>regularization_losses
non_trainable_variables
metrics
?	variables
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
a_
VARIABLE_VALUElstm_18/lstm_cell_18/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%lstm_18/lstm_cell_18/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_18/lstm_cell_18/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEgru_9/gru_cell_9/kernel0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE!gru_9/gru_cell_9/recurrent_kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEgru_9/gru_cell_9/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUElstm_19/lstm_cell_19/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%lstm_19/lstm_cell_19/recurrent_kernel0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_19/lstm_cell_19/bias0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
 
F
0
1
2
3
4
5
6
7
	8

9
 
 

0
1
2

F0
G1
H2
 

F0
G1
H2
В
layer_metrics
layers
 layer_regularization_losses
Ttrainable_variables
Uregularization_losses
non_trainable_variables
metrics
V	variables
 

0
 
 
 
 
 
 
 
 
 

I0
J1
K2
 

I0
J1
K2
В
layer_metrics
layers
 layer_regularization_losses
ctrainable_variables
dregularization_losses
non_trainable_variables
metrics
e	variables
 

0
 
 
 
 

L0
M1
N2
 

L0
M1
N2
В
layer_metrics
layers
 layer_regularization_losses
mtrainable_variables
nregularization_losses
 non_trainable_variables
Ёmetrics
o	variables
 

!0
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
8

Ђtotal

Ѓcount
Є	variables
Ѕ	keras_api
I

Іtotal

Їcount
Ј
_fn_kwargs
Љ	variables
Њ	keras_api
I

Ћtotal

Ќcount
­
_fn_kwargs
Ў	variables
Џ	keras_api
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
Ђ0
Ѓ1

Є	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

І0
Ї1

Љ	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

Ћ0
Ќ1

Ў	variables
~|
VARIABLE_VALUEAdam/dense_27/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_27/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_28/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_28/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_29/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_29/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/lstm_18/lstm_cell_18/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/lstm_18/lstm_cell_18/recurrent_kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/lstm_18/lstm_cell_18/bias/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/gru_9/gru_cell_9/kernel/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(Adam/gru_9/gru_cell_9/recurrent_kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/gru_9/gru_cell_9/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/lstm_19/lstm_cell_19/kernel/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/lstm_19/lstm_cell_19/recurrent_kernel/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/lstm_19/lstm_cell_19/bias/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_27/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_27/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_28/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_28/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_29/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_29/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/lstm_18/lstm_cell_18/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/lstm_18/lstm_cell_18/recurrent_kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/lstm_18/lstm_cell_18/bias/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/gru_9/gru_cell_9/kernel/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(Adam/gru_9/gru_cell_9/recurrent_kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/gru_9/gru_cell_9/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/lstm_19/lstm_cell_19/kernel/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/lstm_19/lstm_cell_19/recurrent_kernel/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/lstm_19/lstm_cell_19/bias/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_10Placeholder*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
dtype0*)
shape :џџџџџџџџџџџџџџџџџџ
Ч
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_10lstm_18/lstm_cell_18/kernel%lstm_18/lstm_cell_18/recurrent_kernellstm_18/lstm_cell_18/biasgru_9/gru_cell_9/biasgru_9/gru_cell_9/kernel!gru_9/gru_cell_9/recurrent_kernellstm_19/lstm_cell_19/kernel%lstm_19/lstm_cell_19/recurrent_kernellstm_19/lstm_cell_19/biasdense_27/kerneldense_27/biasdense_28/kerneldense_28/biasdense_29/kerneldense_29/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*1
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 */
f*R(
&__inference_signature_wrapper_44728724
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
д
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_27/kernel/Read/ReadVariableOp!dense_27/bias/Read/ReadVariableOp#dense_28/kernel/Read/ReadVariableOp!dense_28/bias/Read/ReadVariableOp#dense_29/kernel/Read/ReadVariableOp!dense_29/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/lstm_18/lstm_cell_18/kernel/Read/ReadVariableOp9lstm_18/lstm_cell_18/recurrent_kernel/Read/ReadVariableOp-lstm_18/lstm_cell_18/bias/Read/ReadVariableOp+gru_9/gru_cell_9/kernel/Read/ReadVariableOp5gru_9/gru_cell_9/recurrent_kernel/Read/ReadVariableOp)gru_9/gru_cell_9/bias/Read/ReadVariableOp/lstm_19/lstm_cell_19/kernel/Read/ReadVariableOp9lstm_19/lstm_cell_19/recurrent_kernel/Read/ReadVariableOp-lstm_19/lstm_cell_19/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp*Adam/dense_27/kernel/m/Read/ReadVariableOp(Adam/dense_27/bias/m/Read/ReadVariableOp*Adam/dense_28/kernel/m/Read/ReadVariableOp(Adam/dense_28/bias/m/Read/ReadVariableOp*Adam/dense_29/kernel/m/Read/ReadVariableOp(Adam/dense_29/bias/m/Read/ReadVariableOp6Adam/lstm_18/lstm_cell_18/kernel/m/Read/ReadVariableOp@Adam/lstm_18/lstm_cell_18/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_18/lstm_cell_18/bias/m/Read/ReadVariableOp2Adam/gru_9/gru_cell_9/kernel/m/Read/ReadVariableOp<Adam/gru_9/gru_cell_9/recurrent_kernel/m/Read/ReadVariableOp0Adam/gru_9/gru_cell_9/bias/m/Read/ReadVariableOp6Adam/lstm_19/lstm_cell_19/kernel/m/Read/ReadVariableOp@Adam/lstm_19/lstm_cell_19/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_19/lstm_cell_19/bias/m/Read/ReadVariableOp*Adam/dense_27/kernel/v/Read/ReadVariableOp(Adam/dense_27/bias/v/Read/ReadVariableOp*Adam/dense_28/kernel/v/Read/ReadVariableOp(Adam/dense_28/bias/v/Read/ReadVariableOp*Adam/dense_29/kernel/v/Read/ReadVariableOp(Adam/dense_29/bias/v/Read/ReadVariableOp6Adam/lstm_18/lstm_cell_18/kernel/v/Read/ReadVariableOp@Adam/lstm_18/lstm_cell_18/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_18/lstm_cell_18/bias/v/Read/ReadVariableOp2Adam/gru_9/gru_cell_9/kernel/v/Read/ReadVariableOp<Adam/gru_9/gru_cell_9/recurrent_kernel/v/Read/ReadVariableOp0Adam/gru_9/gru_cell_9/bias/v/Read/ReadVariableOp6Adam/lstm_19/lstm_cell_19/kernel/v/Read/ReadVariableOp@Adam/lstm_19/lstm_cell_19/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_19/lstm_cell_19/bias/v/Read/ReadVariableOpConst*E
Tin>
<2:	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__traced_save_44732387
я
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_27/kerneldense_27/biasdense_28/kerneldense_28/biasdense_29/kerneldense_29/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_18/lstm_cell_18/kernel%lstm_18/lstm_cell_18/recurrent_kernellstm_18/lstm_cell_18/biasgru_9/gru_cell_9/kernel!gru_9/gru_cell_9/recurrent_kernelgru_9/gru_cell_9/biaslstm_19/lstm_cell_19/kernel%lstm_19/lstm_cell_19/recurrent_kernellstm_19/lstm_cell_19/biastotalcounttotal_1count_1total_2count_2Adam/dense_27/kernel/mAdam/dense_27/bias/mAdam/dense_28/kernel/mAdam/dense_28/bias/mAdam/dense_29/kernel/mAdam/dense_29/bias/m"Adam/lstm_18/lstm_cell_18/kernel/m,Adam/lstm_18/lstm_cell_18/recurrent_kernel/m Adam/lstm_18/lstm_cell_18/bias/mAdam/gru_9/gru_cell_9/kernel/m(Adam/gru_9/gru_cell_9/recurrent_kernel/mAdam/gru_9/gru_cell_9/bias/m"Adam/lstm_19/lstm_cell_19/kernel/m,Adam/lstm_19/lstm_cell_19/recurrent_kernel/m Adam/lstm_19/lstm_cell_19/bias/mAdam/dense_27/kernel/vAdam/dense_27/bias/vAdam/dense_28/kernel/vAdam/dense_28/bias/vAdam/dense_29/kernel/vAdam/dense_29/bias/v"Adam/lstm_18/lstm_cell_18/kernel/v,Adam/lstm_18/lstm_cell_18/recurrent_kernel/v Adam/lstm_18/lstm_cell_18/bias/vAdam/gru_9/gru_cell_9/kernel/v(Adam/gru_9/gru_cell_9/recurrent_kernel/vAdam/gru_9/gru_cell_9/bias/v"Adam/lstm_19/lstm_cell_19/kernel/v,Adam/lstm_19/lstm_cell_19/recurrent_kernel/v Adam/lstm_19/lstm_cell_19/bias/v*D
Tin=
;29*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference__traced_restore_44732565 ќ5


*__inference_lstm_19_layer_call_fn_44731789

inputs
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_lstm_19_layer_call_and_return_conditional_losses_447283522
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџK:::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK
 
_user_specified_nameinputs
­
н
J__inference_lstm_cell_19_layer_call_and_return_conditional_losses_44726764

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	KШ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimП
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџ22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
mul_2Ј
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22

IdentityЌ

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_1Ќ

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:џџџџџџџџџK:џџџџџџџџџ2:џџџџџџџџџ2:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџK
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_namestates:OK
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_namestates
Е
Э
while_cond_44726050
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_44726050___redundant_placeholder06
2while_while_cond_44726050___redundant_placeholder16
2while_while_cond_44726050___redundant_placeholder26
2while_while_cond_44726050___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџK:џџџџџџџџџK: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџK:-)
'
_output_shapes
:џџџџџџџџџK:

_output_shapes
: :

_output_shapes
:


#model_9_lstm_18_while_cond_44725105<
8model_9_lstm_18_while_model_9_lstm_18_while_loop_counterB
>model_9_lstm_18_while_model_9_lstm_18_while_maximum_iterations%
!model_9_lstm_18_while_placeholder'
#model_9_lstm_18_while_placeholder_1'
#model_9_lstm_18_while_placeholder_2'
#model_9_lstm_18_while_placeholder_3>
:model_9_lstm_18_while_less_model_9_lstm_18_strided_slice_1V
Rmodel_9_lstm_18_while_model_9_lstm_18_while_cond_44725105___redundant_placeholder0V
Rmodel_9_lstm_18_while_model_9_lstm_18_while_cond_44725105___redundant_placeholder1V
Rmodel_9_lstm_18_while_model_9_lstm_18_while_cond_44725105___redundant_placeholder2V
Rmodel_9_lstm_18_while_model_9_lstm_18_while_cond_44725105___redundant_placeholder3"
model_9_lstm_18_while_identity
Р
model_9/lstm_18/while/LessLess!model_9_lstm_18_while_placeholder:model_9_lstm_18_while_less_model_9_lstm_18_strided_slice_1*
T0*
_output_shapes
: 2
model_9/lstm_18/while/Less
model_9/lstm_18/while/IdentityIdentitymodel_9/lstm_18/while/Less:z:0*
T0
*
_output_shapes
: 2 
model_9/lstm_18/while/Identity"I
model_9_lstm_18_while_identity'model_9/lstm_18/while/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџK:џџџџџџџџџK: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџK:-)
'
_output_shapes
:џџџџџџџџџK:

_output_shapes
: :

_output_shapes
:
O


lstm_19_while_body_44729593,
(lstm_19_while_lstm_19_while_loop_counter2
.lstm_19_while_lstm_19_while_maximum_iterations
lstm_19_while_placeholder
lstm_19_while_placeholder_1
lstm_19_while_placeholder_2
lstm_19_while_placeholder_3+
'lstm_19_while_lstm_19_strided_slice_1_0g
clstm_19_while_tensorarrayv2read_tensorlistgetitem_lstm_19_tensorarrayunstack_tensorlistfromtensor_0?
;lstm_19_while_lstm_cell_19_matmul_readvariableop_resource_0A
=lstm_19_while_lstm_cell_19_matmul_1_readvariableop_resource_0@
<lstm_19_while_lstm_cell_19_biasadd_readvariableop_resource_0
lstm_19_while_identity
lstm_19_while_identity_1
lstm_19_while_identity_2
lstm_19_while_identity_3
lstm_19_while_identity_4
lstm_19_while_identity_5)
%lstm_19_while_lstm_19_strided_slice_1e
alstm_19_while_tensorarrayv2read_tensorlistgetitem_lstm_19_tensorarrayunstack_tensorlistfromtensor=
9lstm_19_while_lstm_cell_19_matmul_readvariableop_resource?
;lstm_19_while_lstm_cell_19_matmul_1_readvariableop_resource>
:lstm_19_while_lstm_cell_19_biasadd_readvariableop_resourceЂ1lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOpЂ0lstm_19/while/lstm_cell_19/MatMul/ReadVariableOpЂ2lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOpг
?lstm_19/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   2A
?lstm_19/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_19/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_19_while_tensorarrayv2read_tensorlistgetitem_lstm_19_tensorarrayunstack_tensorlistfromtensor_0lstm_19_while_placeholderHlstm_19/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџK*
element_dtype023
1lstm_19/while/TensorArrayV2Read/TensorListGetItemс
0lstm_19/while/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp;lstm_19_while_lstm_cell_19_matmul_readvariableop_resource_0*
_output_shapes
:	KШ*
dtype022
0lstm_19/while/lstm_cell_19/MatMul/ReadVariableOpї
!lstm_19/while/lstm_cell_19/MatMulMatMul8lstm_19/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2#
!lstm_19/while/lstm_cell_19/MatMulч
2lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp=lstm_19_while_lstm_cell_19_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype024
2lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOpр
#lstm_19/while/lstm_cell_19/MatMul_1MatMullstm_19_while_placeholder_2:lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2%
#lstm_19/while/lstm_cell_19/MatMul_1и
lstm_19/while/lstm_cell_19/addAddV2+lstm_19/while/lstm_cell_19/MatMul:product:0-lstm_19/while/lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2 
lstm_19/while/lstm_cell_19/addр
1lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp<lstm_19_while_lstm_cell_19_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype023
1lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOpх
"lstm_19/while/lstm_cell_19/BiasAddBiasAdd"lstm_19/while/lstm_cell_19/add:z:09lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2$
"lstm_19/while/lstm_cell_19/BiasAdd
 lstm_19/while/lstm_cell_19/ConstConst*
_output_shapes
: *
dtype0*
value	B :2"
 lstm_19/while/lstm_cell_19/Const
*lstm_19/while/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_19/while/lstm_cell_19/split/split_dimЋ
 lstm_19/while/lstm_cell_19/splitSplit3lstm_19/while/lstm_cell_19/split/split_dim:output:0+lstm_19/while/lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2"
 lstm_19/while/lstm_cell_19/splitА
"lstm_19/while/lstm_cell_19/SigmoidSigmoid)lstm_19/while/lstm_cell_19/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"lstm_19/while/lstm_cell_19/SigmoidД
$lstm_19/while/lstm_cell_19/Sigmoid_1Sigmoid)lstm_19/while/lstm_cell_19/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22&
$lstm_19/while/lstm_cell_19/Sigmoid_1Р
lstm_19/while/lstm_cell_19/mulMul(lstm_19/while/lstm_cell_19/Sigmoid_1:y:0lstm_19_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22 
lstm_19/while/lstm_cell_19/mulЇ
lstm_19/while/lstm_cell_19/ReluRelu)lstm_19/while/lstm_cell_19/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22!
lstm_19/while/lstm_cell_19/Reluд
 lstm_19/while/lstm_cell_19/mul_1Mul&lstm_19/while/lstm_cell_19/Sigmoid:y:0-lstm_19/while/lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22"
 lstm_19/while/lstm_cell_19/mul_1Щ
 lstm_19/while/lstm_cell_19/add_1AddV2"lstm_19/while/lstm_cell_19/mul:z:0$lstm_19/while/lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22"
 lstm_19/while/lstm_cell_19/add_1Д
$lstm_19/while/lstm_cell_19/Sigmoid_2Sigmoid)lstm_19/while/lstm_cell_19/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22&
$lstm_19/while/lstm_cell_19/Sigmoid_2І
!lstm_19/while/lstm_cell_19/Relu_1Relu$lstm_19/while/lstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22#
!lstm_19/while/lstm_cell_19/Relu_1и
 lstm_19/while/lstm_cell_19/mul_2Mul(lstm_19/while/lstm_cell_19/Sigmoid_2:y:0/lstm_19/while/lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22"
 lstm_19/while/lstm_cell_19/mul_2
2lstm_19/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_19_while_placeholder_1lstm_19_while_placeholder$lstm_19/while/lstm_cell_19/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_19/while/TensorArrayV2Write/TensorListSetIteml
lstm_19/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_19/while/add/y
lstm_19/while/addAddV2lstm_19_while_placeholderlstm_19/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_19/while/addp
lstm_19/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_19/while/add_1/y
lstm_19/while/add_1AddV2(lstm_19_while_lstm_19_while_loop_counterlstm_19/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_19/while/add_1
lstm_19/while/IdentityIdentitylstm_19/while/add_1:z:02^lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOp1^lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp3^lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_19/while/Identity­
lstm_19/while/Identity_1Identity.lstm_19_while_lstm_19_while_maximum_iterations2^lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOp1^lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp3^lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_19/while/Identity_1
lstm_19/while/Identity_2Identitylstm_19/while/add:z:02^lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOp1^lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp3^lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_19/while/Identity_2С
lstm_19/while/Identity_3IdentityBlstm_19/while/TensorArrayV2Write/TensorListSetItem:output_handle:02^lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOp1^lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp3^lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_19/while/Identity_3Д
lstm_19/while/Identity_4Identity$lstm_19/while/lstm_cell_19/mul_2:z:02^lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOp1^lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp3^lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_19/while/Identity_4Д
lstm_19/while/Identity_5Identity$lstm_19/while/lstm_cell_19/add_1:z:02^lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOp1^lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp3^lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_19/while/Identity_5"9
lstm_19_while_identitylstm_19/while/Identity:output:0"=
lstm_19_while_identity_1!lstm_19/while/Identity_1:output:0"=
lstm_19_while_identity_2!lstm_19/while/Identity_2:output:0"=
lstm_19_while_identity_3!lstm_19/while/Identity_3:output:0"=
lstm_19_while_identity_4!lstm_19/while/Identity_4:output:0"=
lstm_19_while_identity_5!lstm_19/while/Identity_5:output:0"P
%lstm_19_while_lstm_19_strided_slice_1'lstm_19_while_lstm_19_strided_slice_1_0"z
:lstm_19_while_lstm_cell_19_biasadd_readvariableop_resource<lstm_19_while_lstm_cell_19_biasadd_readvariableop_resource_0"|
;lstm_19_while_lstm_cell_19_matmul_1_readvariableop_resource=lstm_19_while_lstm_cell_19_matmul_1_readvariableop_resource_0"x
9lstm_19_while_lstm_cell_19_matmul_readvariableop_resource;lstm_19_while_lstm_cell_19_matmul_readvariableop_resource_0"Ш
alstm_19_while_tensorarrayv2read_tensorlistgetitem_lstm_19_tensorarrayunstack_tensorlistfromtensorclstm_19_while_tensorarrayv2read_tensorlistgetitem_lstm_19_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : :::2f
1lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOp1lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOp2d
0lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp0lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp2h
2lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOp2lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
Е
п
J__inference_lstm_cell_19_layer_call_and_return_conditional_losses_44732162

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	KШ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimП
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџ22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
mul_2Ј
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22

IdentityЌ

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_1Ќ

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:џџџџџџџџџK:џџџџџџџџџ2:џџџџџџџџџ2:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџK
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ2
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџ2
"
_user_specified_name
states/1
­
н
J__inference_lstm_cell_19_layer_call_and_return_conditional_losses_44726797

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	KШ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimП
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџ22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
mul_2Ј
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22

IdentityЌ

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_1Ќ

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:џџџџџџџџџK:џџџџџџџџџ2:џџџџџџџџџ2:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџK
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_namestates:OK
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_namestates
џZ

#model_9_lstm_18_while_body_44725106<
8model_9_lstm_18_while_model_9_lstm_18_while_loop_counterB
>model_9_lstm_18_while_model_9_lstm_18_while_maximum_iterations%
!model_9_lstm_18_while_placeholder'
#model_9_lstm_18_while_placeholder_1'
#model_9_lstm_18_while_placeholder_2'
#model_9_lstm_18_while_placeholder_3;
7model_9_lstm_18_while_model_9_lstm_18_strided_slice_1_0w
smodel_9_lstm_18_while_tensorarrayv2read_tensorlistgetitem_model_9_lstm_18_tensorarrayunstack_tensorlistfromtensor_0G
Cmodel_9_lstm_18_while_lstm_cell_18_matmul_readvariableop_resource_0I
Emodel_9_lstm_18_while_lstm_cell_18_matmul_1_readvariableop_resource_0H
Dmodel_9_lstm_18_while_lstm_cell_18_biasadd_readvariableop_resource_0"
model_9_lstm_18_while_identity$
 model_9_lstm_18_while_identity_1$
 model_9_lstm_18_while_identity_2$
 model_9_lstm_18_while_identity_3$
 model_9_lstm_18_while_identity_4$
 model_9_lstm_18_while_identity_59
5model_9_lstm_18_while_model_9_lstm_18_strided_slice_1u
qmodel_9_lstm_18_while_tensorarrayv2read_tensorlistgetitem_model_9_lstm_18_tensorarrayunstack_tensorlistfromtensorE
Amodel_9_lstm_18_while_lstm_cell_18_matmul_readvariableop_resourceG
Cmodel_9_lstm_18_while_lstm_cell_18_matmul_1_readvariableop_resourceF
Bmodel_9_lstm_18_while_lstm_cell_18_biasadd_readvariableop_resourceЂ9model_9/lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOpЂ8model_9/lstm_18/while/lstm_cell_18/MatMul/ReadVariableOpЂ:model_9/lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOpу
Gmodel_9/lstm_18/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2I
Gmodel_9/lstm_18/while/TensorArrayV2Read/TensorListGetItem/element_shapeГ
9model_9/lstm_18/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsmodel_9_lstm_18_while_tensorarrayv2read_tensorlistgetitem_model_9_lstm_18_tensorarrayunstack_tensorlistfromtensor_0!model_9_lstm_18_while_placeholderPmodel_9/lstm_18/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02;
9model_9/lstm_18/while/TensorArrayV2Read/TensorListGetItemљ
8model_9/lstm_18/while/lstm_cell_18/MatMul/ReadVariableOpReadVariableOpCmodel_9_lstm_18_while_lstm_cell_18_matmul_readvariableop_resource_0*
_output_shapes
:	Ќ*
dtype02:
8model_9/lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp
)model_9/lstm_18/while/lstm_cell_18/MatMulMatMul@model_9/lstm_18/while/TensorArrayV2Read/TensorListGetItem:item:0@model_9/lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2+
)model_9/lstm_18/while/lstm_cell_18/MatMulџ
:model_9/lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOpEmodel_9_lstm_18_while_lstm_cell_18_matmul_1_readvariableop_resource_0*
_output_shapes
:	KЌ*
dtype02<
:model_9/lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOp
+model_9/lstm_18/while/lstm_cell_18/MatMul_1MatMul#model_9_lstm_18_while_placeholder_2Bmodel_9/lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2-
+model_9/lstm_18/while/lstm_cell_18/MatMul_1ј
&model_9/lstm_18/while/lstm_cell_18/addAddV23model_9/lstm_18/while/lstm_cell_18/MatMul:product:05model_9/lstm_18/while/lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2(
&model_9/lstm_18/while/lstm_cell_18/addј
9model_9/lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOpDmodel_9_lstm_18_while_lstm_cell_18_biasadd_readvariableop_resource_0*
_output_shapes	
:Ќ*
dtype02;
9model_9/lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOp
*model_9/lstm_18/while/lstm_cell_18/BiasAddBiasAdd*model_9/lstm_18/while/lstm_cell_18/add:z:0Amodel_9/lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2,
*model_9/lstm_18/while/lstm_cell_18/BiasAdd
(model_9/lstm_18/while/lstm_cell_18/ConstConst*
_output_shapes
: *
dtype0*
value	B :2*
(model_9/lstm_18/while/lstm_cell_18/ConstЊ
2model_9/lstm_18/while/lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2model_9/lstm_18/while/lstm_cell_18/split/split_dimЫ
(model_9/lstm_18/while/lstm_cell_18/splitSplit;model_9/lstm_18/while/lstm_cell_18/split/split_dim:output:03model_9/lstm_18/while/lstm_cell_18/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*
	num_split2*
(model_9/lstm_18/while/lstm_cell_18/splitШ
*model_9/lstm_18/while/lstm_cell_18/SigmoidSigmoid1model_9/lstm_18/while/lstm_cell_18/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2,
*model_9/lstm_18/while/lstm_cell_18/SigmoidЬ
,model_9/lstm_18/while/lstm_cell_18/Sigmoid_1Sigmoid1model_9/lstm_18/while/lstm_cell_18/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџK2.
,model_9/lstm_18/while/lstm_cell_18/Sigmoid_1р
&model_9/lstm_18/while/lstm_cell_18/mulMul0model_9/lstm_18/while/lstm_cell_18/Sigmoid_1:y:0#model_9_lstm_18_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџK2(
&model_9/lstm_18/while/lstm_cell_18/mulП
'model_9/lstm_18/while/lstm_cell_18/ReluRelu1model_9/lstm_18/while/lstm_cell_18/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџK2)
'model_9/lstm_18/while/lstm_cell_18/Reluє
(model_9/lstm_18/while/lstm_cell_18/mul_1Mul.model_9/lstm_18/while/lstm_cell_18/Sigmoid:y:05model_9/lstm_18/while/lstm_cell_18/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2*
(model_9/lstm_18/while/lstm_cell_18/mul_1щ
(model_9/lstm_18/while/lstm_cell_18/add_1AddV2*model_9/lstm_18/while/lstm_cell_18/mul:z:0,model_9/lstm_18/while/lstm_cell_18/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2*
(model_9/lstm_18/while/lstm_cell_18/add_1Ь
,model_9/lstm_18/while/lstm_cell_18/Sigmoid_2Sigmoid1model_9/lstm_18/while/lstm_cell_18/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџK2.
,model_9/lstm_18/while/lstm_cell_18/Sigmoid_2О
)model_9/lstm_18/while/lstm_cell_18/Relu_1Relu,model_9/lstm_18/while/lstm_cell_18/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2+
)model_9/lstm_18/while/lstm_cell_18/Relu_1ј
(model_9/lstm_18/while/lstm_cell_18/mul_2Mul0model_9/lstm_18/while/lstm_cell_18/Sigmoid_2:y:07model_9/lstm_18/while/lstm_cell_18/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2*
(model_9/lstm_18/while/lstm_cell_18/mul_2А
:model_9/lstm_18/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#model_9_lstm_18_while_placeholder_1!model_9_lstm_18_while_placeholder,model_9/lstm_18/while/lstm_cell_18/mul_2:z:0*
_output_shapes
: *
element_dtype02<
:model_9/lstm_18/while/TensorArrayV2Write/TensorListSetItem|
model_9/lstm_18/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
model_9/lstm_18/while/add/yЉ
model_9/lstm_18/while/addAddV2!model_9_lstm_18_while_placeholder$model_9/lstm_18/while/add/y:output:0*
T0*
_output_shapes
: 2
model_9/lstm_18/while/add
model_9/lstm_18/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
model_9/lstm_18/while/add_1/yЦ
model_9/lstm_18/while/add_1AddV28model_9_lstm_18_while_model_9_lstm_18_while_loop_counter&model_9/lstm_18/while/add_1/y:output:0*
T0*
_output_shapes
: 2
model_9/lstm_18/while/add_1Т
model_9/lstm_18/while/IdentityIdentitymodel_9/lstm_18/while/add_1:z:0:^model_9/lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOp9^model_9/lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp;^model_9/lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2 
model_9/lstm_18/while/Identityх
 model_9/lstm_18/while/Identity_1Identity>model_9_lstm_18_while_model_9_lstm_18_while_maximum_iterations:^model_9/lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOp9^model_9/lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp;^model_9/lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2"
 model_9/lstm_18/while/Identity_1Ф
 model_9/lstm_18/while/Identity_2Identitymodel_9/lstm_18/while/add:z:0:^model_9/lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOp9^model_9/lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp;^model_9/lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2"
 model_9/lstm_18/while/Identity_2ё
 model_9/lstm_18/while/Identity_3IdentityJmodel_9/lstm_18/while/TensorArrayV2Write/TensorListSetItem:output_handle:0:^model_9/lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOp9^model_9/lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp;^model_9/lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2"
 model_9/lstm_18/while/Identity_3ф
 model_9/lstm_18/while/Identity_4Identity,model_9/lstm_18/while/lstm_cell_18/mul_2:z:0:^model_9/lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOp9^model_9/lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp;^model_9/lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2"
 model_9/lstm_18/while/Identity_4ф
 model_9/lstm_18/while/Identity_5Identity,model_9/lstm_18/while/lstm_cell_18/add_1:z:0:^model_9/lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOp9^model_9/lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp;^model_9/lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2"
 model_9/lstm_18/while/Identity_5"I
model_9_lstm_18_while_identity'model_9/lstm_18/while/Identity:output:0"M
 model_9_lstm_18_while_identity_1)model_9/lstm_18/while/Identity_1:output:0"M
 model_9_lstm_18_while_identity_2)model_9/lstm_18/while/Identity_2:output:0"M
 model_9_lstm_18_while_identity_3)model_9/lstm_18/while/Identity_3:output:0"M
 model_9_lstm_18_while_identity_4)model_9/lstm_18/while/Identity_4:output:0"M
 model_9_lstm_18_while_identity_5)model_9/lstm_18/while/Identity_5:output:0"
Bmodel_9_lstm_18_while_lstm_cell_18_biasadd_readvariableop_resourceDmodel_9_lstm_18_while_lstm_cell_18_biasadd_readvariableop_resource_0"
Cmodel_9_lstm_18_while_lstm_cell_18_matmul_1_readvariableop_resourceEmodel_9_lstm_18_while_lstm_cell_18_matmul_1_readvariableop_resource_0"
Amodel_9_lstm_18_while_lstm_cell_18_matmul_readvariableop_resourceCmodel_9_lstm_18_while_lstm_cell_18_matmul_readvariableop_resource_0"p
5model_9_lstm_18_while_model_9_lstm_18_strided_slice_17model_9_lstm_18_while_model_9_lstm_18_strided_slice_1_0"ш
qmodel_9_lstm_18_while_tensorarrayv2read_tensorlistgetitem_model_9_lstm_18_tensorarrayunstack_tensorlistfromtensorsmodel_9_lstm_18_while_tensorarrayv2read_tensorlistgetitem_model_9_lstm_18_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџK:џџџџџџџџџK: : :::2v
9model_9/lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOp9model_9/lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOp2t
8model_9/lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp8model_9/lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp2x
:model_9/lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOp:model_9/lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџK:-)
'
_output_shapes
:џџџџџџџџџK:

_output_shapes
: :

_output_shapes
: 
З
u
K__inference_concatenate_9_layer_call_and_return_conditional_losses_44728443

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ`2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:џџџџџџџџџ :џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ж

*__inference_lstm_18_layer_call_fn_44730098

inputs
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_lstm_18_layer_call_and_return_conditional_losses_447276102
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ёZ
и
C__inference_gru_9_layer_call_and_return_conditional_losses_44730952
inputs_0&
"gru_cell_9_readvariableop_resource-
)gru_cell_9_matmul_readvariableop_resource/
+gru_cell_9_matmul_1_readvariableop_resource
identityЂ gru_cell_9/MatMul/ReadVariableOpЂ"gru_cell_9/MatMul_1/ReadVariableOpЂgru_cell_9/ReadVariableOpЂwhileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2
gru_cell_9/ReadVariableOpReadVariableOp"gru_cell_9_readvariableop_resource*
_output_shapes
:	*
dtype02
gru_cell_9/ReadVariableOp
gru_cell_9/unstackUnpack!gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
gru_cell_9/unstackЏ
 gru_cell_9/MatMul/ReadVariableOpReadVariableOp)gru_cell_9_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 gru_cell_9/MatMul/ReadVariableOpЇ
gru_cell_9/MatMulMatMulstrided_slice_2:output:0(gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_cell_9/MatMul 
gru_cell_9/BiasAddBiasAddgru_cell_9/MatMul:product:0gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_cell_9/BiasAddf
gru_cell_9/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_9/Const
gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_9/split/split_dimи
gru_cell_9/splitSplit#gru_cell_9/split/split_dim:output:0gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
gru_cell_9/splitЕ
"gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_9_matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype02$
"gru_cell_9/MatMul_1/ReadVariableOpЃ
gru_cell_9/MatMul_1MatMulzeros:output:0*gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_cell_9/MatMul_1І
gru_cell_9/BiasAdd_1BiasAddgru_cell_9/MatMul_1:product:0gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_cell_9/BiasAdd_1}
gru_cell_9/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   џџџџ2
gru_cell_9/Const_1
gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_9/split_1/split_dim
gru_cell_9/split_1SplitVgru_cell_9/BiasAdd_1:output:0gru_cell_9/Const_1:output:0%gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
gru_cell_9/split_1
gru_cell_9/addAddV2gru_cell_9/split:output:0gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/addy
gru_cell_9/SigmoidSigmoidgru_cell_9/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/Sigmoid
gru_cell_9/add_1AddV2gru_cell_9/split:output:1gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/add_1
gru_cell_9/Sigmoid_1Sigmoidgru_cell_9/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/Sigmoid_1
gru_cell_9/mulMulgru_cell_9/Sigmoid_1:y:0gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/mul
gru_cell_9/add_2AddV2gru_cell_9/split:output:2gru_cell_9/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/add_2r
gru_cell_9/ReluRelugru_cell_9/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/Relu
gru_cell_9/mul_1Mulgru_cell_9/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/mul_1i
gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_9/sub/x
gru_cell_9/subSubgru_cell_9/sub/x:output:0gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/sub
gru_cell_9/mul_2Mulgru_cell_9/sub:z:0gru_cell_9/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/mul_2
gru_cell_9/add_3AddV2gru_cell_9/mul_1:z:0gru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЋ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_9_readvariableop_resource)gru_cell_9_matmul_readvariableop_resource+gru_cell_9_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_44730862*
condR
while_cond_44730861*8
output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeи
IdentityIdentitystrided_slice_3:output:0!^gru_cell_9/MatMul/ReadVariableOp#^gru_cell_9/MatMul_1/ReadVariableOp^gru_cell_9/ReadVariableOp^while*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2D
 gru_cell_9/MatMul/ReadVariableOp gru_cell_9/MatMul/ReadVariableOp2H
"gru_cell_9/MatMul_1/ReadVariableOp"gru_cell_9/MatMul_1/ReadVariableOp26
gru_cell_9/ReadVariableOpgru_cell_9/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
я
g
H__inference_dropout_18_layer_call_and_return_conditional_losses_44727999

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЂМ?2
dropout/Const
dropout/MulMulinputsdropout/Const:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeС
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL=2
dropout/GreaterEqual/yЫ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
dropout/Mul_1r
IdentityIdentitydropout/Mul_1:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :џџџџџџџџџџџџџџџџџџK:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK
 
_user_specified_nameinputs
я
п
$__inference__traced_restore_44732565
file_prefix$
 assignvariableop_dense_27_kernel$
 assignvariableop_1_dense_27_bias&
"assignvariableop_2_dense_28_kernel$
 assignvariableop_3_dense_28_bias&
"assignvariableop_4_dense_29_kernel$
 assignvariableop_5_dense_29_bias 
assignvariableop_6_adam_iter"
assignvariableop_7_adam_beta_1"
assignvariableop_8_adam_beta_2!
assignvariableop_9_adam_decay*
&assignvariableop_10_adam_learning_rate3
/assignvariableop_11_lstm_18_lstm_cell_18_kernel=
9assignvariableop_12_lstm_18_lstm_cell_18_recurrent_kernel1
-assignvariableop_13_lstm_18_lstm_cell_18_bias/
+assignvariableop_14_gru_9_gru_cell_9_kernel9
5assignvariableop_15_gru_9_gru_cell_9_recurrent_kernel-
)assignvariableop_16_gru_9_gru_cell_9_bias3
/assignvariableop_17_lstm_19_lstm_cell_19_kernel=
9assignvariableop_18_lstm_19_lstm_cell_19_recurrent_kernel1
-assignvariableop_19_lstm_19_lstm_cell_19_bias
assignvariableop_20_total
assignvariableop_21_count
assignvariableop_22_total_1
assignvariableop_23_count_1
assignvariableop_24_total_2
assignvariableop_25_count_2.
*assignvariableop_26_adam_dense_27_kernel_m,
(assignvariableop_27_adam_dense_27_bias_m.
*assignvariableop_28_adam_dense_28_kernel_m,
(assignvariableop_29_adam_dense_28_bias_m.
*assignvariableop_30_adam_dense_29_kernel_m,
(assignvariableop_31_adam_dense_29_bias_m:
6assignvariableop_32_adam_lstm_18_lstm_cell_18_kernel_mD
@assignvariableop_33_adam_lstm_18_lstm_cell_18_recurrent_kernel_m8
4assignvariableop_34_adam_lstm_18_lstm_cell_18_bias_m6
2assignvariableop_35_adam_gru_9_gru_cell_9_kernel_m@
<assignvariableop_36_adam_gru_9_gru_cell_9_recurrent_kernel_m4
0assignvariableop_37_adam_gru_9_gru_cell_9_bias_m:
6assignvariableop_38_adam_lstm_19_lstm_cell_19_kernel_mD
@assignvariableop_39_adam_lstm_19_lstm_cell_19_recurrent_kernel_m8
4assignvariableop_40_adam_lstm_19_lstm_cell_19_bias_m.
*assignvariableop_41_adam_dense_27_kernel_v,
(assignvariableop_42_adam_dense_27_bias_v.
*assignvariableop_43_adam_dense_28_kernel_v,
(assignvariableop_44_adam_dense_28_bias_v.
*assignvariableop_45_adam_dense_29_kernel_v,
(assignvariableop_46_adam_dense_29_bias_v:
6assignvariableop_47_adam_lstm_18_lstm_cell_18_kernel_vD
@assignvariableop_48_adam_lstm_18_lstm_cell_18_recurrent_kernel_v8
4assignvariableop_49_adam_lstm_18_lstm_cell_18_bias_v6
2assignvariableop_50_adam_gru_9_gru_cell_9_kernel_v@
<assignvariableop_51_adam_gru_9_gru_cell_9_recurrent_kernel_v4
0assignvariableop_52_adam_gru_9_gru_cell_9_bias_v:
6assignvariableop_53_adam_lstm_19_lstm_cell_19_kernel_vD
@assignvariableop_54_adam_lstm_19_lstm_cell_19_recurrent_kernel_v8
4assignvariableop_55_adam_lstm_19_lstm_cell_19_bias_v
identity_57ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_40ЂAssignVariableOp_41ЂAssignVariableOp_42ЂAssignVariableOp_43ЂAssignVariableOp_44ЂAssignVariableOp_45ЂAssignVariableOp_46ЂAssignVariableOp_47ЂAssignVariableOp_48ЂAssignVariableOp_49ЂAssignVariableOp_5ЂAssignVariableOp_50ЂAssignVariableOp_51ЂAssignVariableOp_52ЂAssignVariableOp_53ЂAssignVariableOp_54ЂAssignVariableOp_55ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9М
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:9*
dtype0*Ш
valueОBЛ9B6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:9*
dtype0*
value|Bz9B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesЫ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*њ
_output_shapesч
ф:::::::::::::::::::::::::::::::::::::::::::::::::::::::::*G
dtypes=
;29	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp assignvariableop_dense_27_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ѕ
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_27_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ї
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_28_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ѕ
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_28_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Ї
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_29_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ѕ
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_29_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6Ё
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ѓ
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Ѓ
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Ђ
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ў
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11З
AssignVariableOp_11AssignVariableOp/assignvariableop_11_lstm_18_lstm_cell_18_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12С
AssignVariableOp_12AssignVariableOp9assignvariableop_12_lstm_18_lstm_cell_18_recurrent_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Е
AssignVariableOp_13AssignVariableOp-assignvariableop_13_lstm_18_lstm_cell_18_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Г
AssignVariableOp_14AssignVariableOp+assignvariableop_14_gru_9_gru_cell_9_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Н
AssignVariableOp_15AssignVariableOp5assignvariableop_15_gru_9_gru_cell_9_recurrent_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Б
AssignVariableOp_16AssignVariableOp)assignvariableop_16_gru_9_gru_cell_9_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17З
AssignVariableOp_17AssignVariableOp/assignvariableop_17_lstm_19_lstm_cell_19_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18С
AssignVariableOp_18AssignVariableOp9assignvariableop_18_lstm_19_lstm_cell_19_recurrent_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Е
AssignVariableOp_19AssignVariableOp-assignvariableop_19_lstm_19_lstm_cell_19_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Ё
AssignVariableOp_20AssignVariableOpassignvariableop_20_totalIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Ё
AssignVariableOp_21AssignVariableOpassignvariableop_21_countIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Ѓ
AssignVariableOp_22AssignVariableOpassignvariableop_22_total_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Ѓ
AssignVariableOp_23AssignVariableOpassignvariableop_23_count_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Ѓ
AssignVariableOp_24AssignVariableOpassignvariableop_24_total_2Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Ѓ
AssignVariableOp_25AssignVariableOpassignvariableop_25_count_2Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26В
AssignVariableOp_26AssignVariableOp*assignvariableop_26_adam_dense_27_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27А
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_dense_27_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28В
AssignVariableOp_28AssignVariableOp*assignvariableop_28_adam_dense_28_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29А
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_dense_28_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30В
AssignVariableOp_30AssignVariableOp*assignvariableop_30_adam_dense_29_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31А
AssignVariableOp_31AssignVariableOp(assignvariableop_31_adam_dense_29_bias_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32О
AssignVariableOp_32AssignVariableOp6assignvariableop_32_adam_lstm_18_lstm_cell_18_kernel_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33Ш
AssignVariableOp_33AssignVariableOp@assignvariableop_33_adam_lstm_18_lstm_cell_18_recurrent_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34М
AssignVariableOp_34AssignVariableOp4assignvariableop_34_adam_lstm_18_lstm_cell_18_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35К
AssignVariableOp_35AssignVariableOp2assignvariableop_35_adam_gru_9_gru_cell_9_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36Ф
AssignVariableOp_36AssignVariableOp<assignvariableop_36_adam_gru_9_gru_cell_9_recurrent_kernel_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37И
AssignVariableOp_37AssignVariableOp0assignvariableop_37_adam_gru_9_gru_cell_9_bias_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38О
AssignVariableOp_38AssignVariableOp6assignvariableop_38_adam_lstm_19_lstm_cell_19_kernel_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39Ш
AssignVariableOp_39AssignVariableOp@assignvariableop_39_adam_lstm_19_lstm_cell_19_recurrent_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40М
AssignVariableOp_40AssignVariableOp4assignvariableop_40_adam_lstm_19_lstm_cell_19_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41В
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_dense_27_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42А
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_dense_27_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43В
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_dense_28_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44А
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_dense_28_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45В
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_dense_29_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46А
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_dense_29_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47О
AssignVariableOp_47AssignVariableOp6assignvariableop_47_adam_lstm_18_lstm_cell_18_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48Ш
AssignVariableOp_48AssignVariableOp@assignvariableop_48_adam_lstm_18_lstm_cell_18_recurrent_kernel_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49М
AssignVariableOp_49AssignVariableOp4assignvariableop_49_adam_lstm_18_lstm_cell_18_bias_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50К
AssignVariableOp_50AssignVariableOp2assignvariableop_50_adam_gru_9_gru_cell_9_kernel_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51Ф
AssignVariableOp_51AssignVariableOp<assignvariableop_51_adam_gru_9_gru_cell_9_recurrent_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52И
AssignVariableOp_52AssignVariableOp0assignvariableop_52_adam_gru_9_gru_cell_9_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53О
AssignVariableOp_53AssignVariableOp6assignvariableop_53_adam_lstm_19_lstm_cell_19_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54Ш
AssignVariableOp_54AssignVariableOp@assignvariableop_54_adam_lstm_19_lstm_cell_19_recurrent_kernel_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55М
AssignVariableOp_55AssignVariableOp4assignvariableop_55_adam_lstm_19_lstm_cell_19_bias_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_559
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpЎ

Identity_56Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_56Ё

Identity_57IdentityIdentity_56:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_57"#
identity_57Identity_57:output:0*ї
_input_shapesх
т: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_55AssignVariableOp_552(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Щ[
є
E__inference_lstm_18_layer_call_and_return_conditional_losses_44730076

inputs/
+lstm_cell_18_matmul_readvariableop_resource1
-lstm_cell_18_matmul_1_readvariableop_resource0
,lstm_cell_18_biasadd_readvariableop_resource
identityЂ#lstm_cell_18/BiasAdd/ReadVariableOpЂ"lstm_cell_18/MatMul/ReadVariableOpЂ$lstm_cell_18/MatMul_1/ReadVariableOpЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2Е
"lstm_cell_18/MatMul/ReadVariableOpReadVariableOp+lstm_cell_18_matmul_readvariableop_resource*
_output_shapes
:	Ќ*
dtype02$
"lstm_cell_18/MatMul/ReadVariableOp­
lstm_cell_18/MatMulMatMulstrided_slice_2:output:0*lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_18/MatMulЛ
$lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_18_matmul_1_readvariableop_resource*
_output_shapes
:	KЌ*
dtype02&
$lstm_cell_18/MatMul_1/ReadVariableOpЉ
lstm_cell_18/MatMul_1MatMulzeros:output:0,lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_18/MatMul_1 
lstm_cell_18/addAddV2lstm_cell_18/MatMul:product:0lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_18/addД
#lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_18_biasadd_readvariableop_resource*
_output_shapes	
:Ќ*
dtype02%
#lstm_cell_18/BiasAdd/ReadVariableOp­
lstm_cell_18/BiasAddBiasAddlstm_cell_18/add:z:0+lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_18/BiasAddj
lstm_cell_18/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_18/Const~
lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_18/split/split_dimѓ
lstm_cell_18/splitSplit%lstm_cell_18/split/split_dim:output:0lstm_cell_18/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*
	num_split2
lstm_cell_18/split
lstm_cell_18/SigmoidSigmoidlstm_cell_18/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/Sigmoid
lstm_cell_18/Sigmoid_1Sigmoidlstm_cell_18/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/Sigmoid_1
lstm_cell_18/mulMullstm_cell_18/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/mul}
lstm_cell_18/ReluRelulstm_cell_18/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/Relu
lstm_cell_18/mul_1Mullstm_cell_18/Sigmoid:y:0lstm_cell_18/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/mul_1
lstm_cell_18/add_1AddV2lstm_cell_18/mul:z:0lstm_cell_18/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/add_1
lstm_cell_18/Sigmoid_2Sigmoidlstm_cell_18/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/Sigmoid_2|
lstm_cell_18/Relu_1Relulstm_cell_18/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/Relu_1 
lstm_cell_18/mul_2Mullstm_cell_18/Sigmoid_2:y:0!lstm_cell_18/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterђ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_18_matmul_readvariableop_resource-lstm_cell_18_matmul_1_readvariableop_resource,lstm_cell_18_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџK:џџџџџџџџџK: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_44729991*
condR
while_cond_44729990*K
output_shapes:
8: : : : :џџџџџџџџџK:џџџџџџџџџK: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџK*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeъ
IdentityIdentitytranspose_1:y:0$^lstm_cell_18/BiasAdd/ReadVariableOp#^lstm_cell_18/MatMul/ReadVariableOp%^lstm_cell_18/MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2J
#lstm_cell_18/BiasAdd/ReadVariableOp#lstm_cell_18/BiasAdd/ReadVariableOp2H
"lstm_cell_18/MatMul/ReadVariableOp"lstm_cell_18/MatMul/ReadVariableOp2L
$lstm_cell_18/MatMul_1/ReadVariableOp$lstm_cell_18/MatMul_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

I
-__inference_dropout_19_layer_call_fn_44731816

inputs
identityЩ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dropout_19_layer_call_and_return_conditional_losses_447280342
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ2:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
я
g
H__inference_dropout_18_layer_call_and_return_conditional_losses_44730438

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЂМ?2
dropout/Const
dropout/MulMulinputsdropout/Const:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeС
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL=2
dropout/GreaterEqual/yЫ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
dropout/Mul_1r
IdentityIdentitydropout/Mul_1:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :џџџџџџџџџџџџџџџџџџK:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK
 
_user_specified_nameinputs
џF
Ї
while_body_44730862
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
*while_gru_cell_9_readvariableop_resource_05
1while_gru_cell_9_matmul_readvariableop_resource_07
3while_gru_cell_9_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
(while_gru_cell_9_readvariableop_resource3
/while_gru_cell_9_matmul_readvariableop_resource5
1while_gru_cell_9_matmul_1_readvariableop_resourceЂ&while/gru_cell_9/MatMul/ReadVariableOpЂ(while/gru_cell_9/MatMul_1/ReadVariableOpЂwhile/gru_cell_9/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЎ
while/gru_cell_9/ReadVariableOpReadVariableOp*while_gru_cell_9_readvariableop_resource_0*
_output_shapes
:	*
dtype02!
while/gru_cell_9/ReadVariableOp
while/gru_cell_9/unstackUnpack'while/gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
while/gru_cell_9/unstackУ
&while/gru_cell_9/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/gru_cell_9/MatMul/ReadVariableOpб
while/gru_cell_9/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
while/gru_cell_9/MatMulИ
while/gru_cell_9/BiasAddBiasAdd!while/gru_cell_9/MatMul:product:0!while/gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
while/gru_cell_9/BiasAddr
while/gru_cell_9/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_9/Const
 while/gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2"
 while/gru_cell_9/split/split_dim№
while/gru_cell_9/splitSplit)while/gru_cell_9/split/split_dim:output:0!while/gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/gru_cell_9/splitЩ
(while/gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes
:	2*
dtype02*
(while/gru_cell_9/MatMul_1/ReadVariableOpК
while/gru_cell_9/MatMul_1MatMulwhile_placeholder_20while/gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
while/gru_cell_9/MatMul_1О
while/gru_cell_9/BiasAdd_1BiasAdd#while/gru_cell_9/MatMul_1:product:0!while/gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2
while/gru_cell_9/BiasAdd_1
while/gru_cell_9/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   џџџџ2
while/gru_cell_9/Const_1
"while/gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2$
"while/gru_cell_9/split_1/split_dimЈ
while/gru_cell_9/split_1SplitV#while/gru_cell_9/BiasAdd_1:output:0!while/gru_cell_9/Const_1:output:0+while/gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/gru_cell_9/split_1Ћ
while/gru_cell_9/addAddV2while/gru_cell_9/split:output:0!while/gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/add
while/gru_cell_9/SigmoidSigmoidwhile/gru_cell_9/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/SigmoidЏ
while/gru_cell_9/add_1AddV2while/gru_cell_9/split:output:1!while/gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/add_1
while/gru_cell_9/Sigmoid_1Sigmoidwhile/gru_cell_9/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/Sigmoid_1Ј
while/gru_cell_9/mulMulwhile/gru_cell_9/Sigmoid_1:y:0!while/gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/mulІ
while/gru_cell_9/add_2AddV2while/gru_cell_9/split:output:2while/gru_cell_9/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/add_2
while/gru_cell_9/ReluReluwhile/gru_cell_9/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/Relu
while/gru_cell_9/mul_1Mulwhile/gru_cell_9/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/mul_1u
while/gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_9/sub/xЄ
while/gru_cell_9/subSubwhile/gru_cell_9/sub/x:output:0while/gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/subЈ
while/gru_cell_9/mul_2Mulwhile/gru_cell_9/sub:z:0#while/gru_cell_9/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/mul_2Ѓ
while/gru_cell_9/add_3AddV2while/gru_cell_9/mul_1:z:0while/gru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/add_3о
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_9/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1д
while/IdentityIdentitywhile/add_1:z:0'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityч
while/Identity_1Identitywhile_while_maximum_iterations'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1ж
while/Identity_2Identitywhile/add:z:0'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3є
while/Identity_4Identitywhile/gru_cell_9/add_3:z:0'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4"h
1while_gru_cell_9_matmul_1_readvariableop_resource3while_gru_cell_9_matmul_1_readvariableop_resource_0"d
/while_gru_cell_9_matmul_readvariableop_resource1while_gru_cell_9_matmul_readvariableop_resource_0"V
(while_gru_cell_9_readvariableop_resource*while_gru_cell_9_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ2: : :::2P
&while/gru_cell_9/MatMul/ReadVariableOp&while/gru_cell_9/MatMul/ReadVariableOp2T
(while/gru_cell_9/MatMul_1/ReadVariableOp(while/gru_cell_9/MatMul_1/ReadVariableOp2B
while/gru_cell_9/ReadVariableOpwhile/gru_cell_9/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
у	
Џ
-__inference_gru_cell_9_layer_call_fn_44732096

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1ЂStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ2:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_gru_cell_9_layer_call_and_return_conditional_losses_447262412
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ2:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ2
"
_user_specified_name
states/0
к
Д
while_cond_44731020
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_44731020___redundant_placeholder06
2while_while_cond_44731020___redundant_placeholder16
2while_while_cond_44731020___redundant_placeholder26
2while_while_cond_44731020___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
C

while_body_44730319
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_18_matmul_readvariableop_resource_09
5while_lstm_cell_18_matmul_1_readvariableop_resource_08
4while_lstm_cell_18_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_18_matmul_readvariableop_resource7
3while_lstm_cell_18_matmul_1_readvariableop_resource6
2while_lstm_cell_18_biasadd_readvariableop_resourceЂ)while/lstm_cell_18/BiasAdd/ReadVariableOpЂ(while/lstm_cell_18/MatMul/ReadVariableOpЂ*while/lstm_cell_18/MatMul_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЩ
(while/lstm_cell_18/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_18_matmul_readvariableop_resource_0*
_output_shapes
:	Ќ*
dtype02*
(while/lstm_cell_18/MatMul/ReadVariableOpз
while/lstm_cell_18/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_18/MatMulЯ
*while/lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_18_matmul_1_readvariableop_resource_0*
_output_shapes
:	KЌ*
dtype02,
*while/lstm_cell_18/MatMul_1/ReadVariableOpР
while/lstm_cell_18/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_18/MatMul_1И
while/lstm_cell_18/addAddV2#while/lstm_cell_18/MatMul:product:0%while/lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_18/addШ
)while/lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_18_biasadd_readvariableop_resource_0*
_output_shapes	
:Ќ*
dtype02+
)while/lstm_cell_18/BiasAdd/ReadVariableOpХ
while/lstm_cell_18/BiasAddBiasAddwhile/lstm_cell_18/add:z:01while/lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_18/BiasAddv
while/lstm_cell_18/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_18/Const
"while/lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_18/split/split_dim
while/lstm_cell_18/splitSplit+while/lstm_cell_18/split/split_dim:output:0#while/lstm_cell_18/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*
	num_split2
while/lstm_cell_18/split
while/lstm_cell_18/SigmoidSigmoid!while/lstm_cell_18/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/Sigmoid
while/lstm_cell_18/Sigmoid_1Sigmoid!while/lstm_cell_18/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/Sigmoid_1 
while/lstm_cell_18/mulMul while/lstm_cell_18/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/mul
while/lstm_cell_18/ReluRelu!while/lstm_cell_18/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/ReluД
while/lstm_cell_18/mul_1Mulwhile/lstm_cell_18/Sigmoid:y:0%while/lstm_cell_18/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/mul_1Љ
while/lstm_cell_18/add_1AddV2while/lstm_cell_18/mul:z:0while/lstm_cell_18/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/add_1
while/lstm_cell_18/Sigmoid_2Sigmoid!while/lstm_cell_18/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/Sigmoid_2
while/lstm_cell_18/Relu_1Reluwhile/lstm_cell_18/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/Relu_1И
while/lstm_cell_18/mul_2Mul while/lstm_cell_18/Sigmoid_2:y:0'while/lstm_cell_18/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_18/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1т
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_18/BiasAdd/ReadVariableOp)^while/lstm_cell_18/MatMul/ReadVariableOp+^while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityѕ
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_18/BiasAdd/ReadVariableOp)^while/lstm_cell_18/MatMul/ReadVariableOp+^while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1ф
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_18/BiasAdd/ReadVariableOp)^while/lstm_cell_18/MatMul/ReadVariableOp+^while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_18/BiasAdd/ReadVariableOp)^while/lstm_cell_18/MatMul/ReadVariableOp+^while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_18/mul_2:z:0*^while/lstm_cell_18/BiasAdd/ReadVariableOp)^while/lstm_cell_18/MatMul/ReadVariableOp+^while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_18/add_1:z:0*^while/lstm_cell_18/BiasAdd/ReadVariableOp)^while/lstm_cell_18/MatMul/ReadVariableOp+^while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_18_biasadd_readvariableop_resource4while_lstm_cell_18_biasadd_readvariableop_resource_0"l
3while_lstm_cell_18_matmul_1_readvariableop_resource5while_lstm_cell_18_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_18_matmul_readvariableop_resource3while_lstm_cell_18_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџK:џџџџџџџџџK: : :::2V
)while/lstm_cell_18/BiasAdd/ReadVariableOp)while/lstm_cell_18/BiasAdd/ReadVariableOp2T
(while/lstm_cell_18/MatMul/ReadVariableOp(while/lstm_cell_18/MatMul/ReadVariableOp2X
*while/lstm_cell_18/MatMul_1/ReadVariableOp*while/lstm_cell_18/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџK:-)
'
_output_shapes
:џџџџџџџџџK:

_output_shapes
: :

_output_shapes
: 
Ќе
ь
E__inference_model_9_layer_call_and_return_conditional_losses_44729700

inputs7
3lstm_18_lstm_cell_18_matmul_readvariableop_resource9
5lstm_18_lstm_cell_18_matmul_1_readvariableop_resource8
4lstm_18_lstm_cell_18_biasadd_readvariableop_resource,
(gru_9_gru_cell_9_readvariableop_resource3
/gru_9_gru_cell_9_matmul_readvariableop_resource5
1gru_9_gru_cell_9_matmul_1_readvariableop_resource7
3lstm_19_lstm_cell_19_matmul_readvariableop_resource9
5lstm_19_lstm_cell_19_matmul_1_readvariableop_resource8
4lstm_19_lstm_cell_19_biasadd_readvariableop_resource+
'dense_27_matmul_readvariableop_resource,
(dense_27_biasadd_readvariableop_resource+
'dense_28_matmul_readvariableop_resource,
(dense_28_biasadd_readvariableop_resource+
'dense_29_matmul_readvariableop_resource,
(dense_29_biasadd_readvariableop_resource
identityЂdense_27/BiasAdd/ReadVariableOpЂdense_27/MatMul/ReadVariableOpЂdense_28/BiasAdd/ReadVariableOpЂdense_28/MatMul/ReadVariableOpЂdense_29/BiasAdd/ReadVariableOpЂdense_29/MatMul/ReadVariableOpЂ&gru_9/gru_cell_9/MatMul/ReadVariableOpЂ(gru_9/gru_cell_9/MatMul_1/ReadVariableOpЂgru_9/gru_cell_9/ReadVariableOpЂgru_9/whileЂ+lstm_18/lstm_cell_18/BiasAdd/ReadVariableOpЂ*lstm_18/lstm_cell_18/MatMul/ReadVariableOpЂ,lstm_18/lstm_cell_18/MatMul_1/ReadVariableOpЂlstm_18/whileЂ+lstm_19/lstm_cell_19/BiasAdd/ReadVariableOpЂ*lstm_19/lstm_cell_19/MatMul/ReadVariableOpЂ,lstm_19/lstm_cell_19/MatMul_1/ReadVariableOpЂlstm_19/whileT
lstm_18/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_18/Shape
lstm_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_18/strided_slice/stack
lstm_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_18/strided_slice/stack_1
lstm_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_18/strided_slice/stack_2
lstm_18/strided_sliceStridedSlicelstm_18/Shape:output:0$lstm_18/strided_slice/stack:output:0&lstm_18/strided_slice/stack_1:output:0&lstm_18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_18/strided_slicel
lstm_18/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
lstm_18/zeros/mul/y
lstm_18/zeros/mulMullstm_18/strided_slice:output:0lstm_18/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_18/zeros/mulo
lstm_18/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_18/zeros/Less/y
lstm_18/zeros/LessLesslstm_18/zeros/mul:z:0lstm_18/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_18/zeros/Lessr
lstm_18/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
lstm_18/zeros/packed/1Ѓ
lstm_18/zeros/packedPacklstm_18/strided_slice:output:0lstm_18/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_18/zeros/packedo
lstm_18/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_18/zeros/Const
lstm_18/zerosFilllstm_18/zeros/packed:output:0lstm_18/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_18/zerosp
lstm_18/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
lstm_18/zeros_1/mul/y
lstm_18/zeros_1/mulMullstm_18/strided_slice:output:0lstm_18/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_18/zeros_1/muls
lstm_18/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_18/zeros_1/Less/y
lstm_18/zeros_1/LessLesslstm_18/zeros_1/mul:z:0lstm_18/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_18/zeros_1/Lessv
lstm_18/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
lstm_18/zeros_1/packed/1Љ
lstm_18/zeros_1/packedPacklstm_18/strided_slice:output:0!lstm_18/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_18/zeros_1/packeds
lstm_18/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_18/zeros_1/Const
lstm_18/zeros_1Filllstm_18/zeros_1/packed:output:0lstm_18/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_18/zeros_1
lstm_18/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_18/transpose/perm
lstm_18/transpose	Transposeinputslstm_18/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
lstm_18/transposeg
lstm_18/Shape_1Shapelstm_18/transpose:y:0*
T0*
_output_shapes
:2
lstm_18/Shape_1
lstm_18/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_18/strided_slice_1/stack
lstm_18/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_18/strided_slice_1/stack_1
lstm_18/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_18/strided_slice_1/stack_2
lstm_18/strided_slice_1StridedSlicelstm_18/Shape_1:output:0&lstm_18/strided_slice_1/stack:output:0(lstm_18/strided_slice_1/stack_1:output:0(lstm_18/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_18/strided_slice_1
#lstm_18/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#lstm_18/TensorArrayV2/element_shapeв
lstm_18/TensorArrayV2TensorListReserve,lstm_18/TensorArrayV2/element_shape:output:0 lstm_18/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_18/TensorArrayV2Я
=lstm_18/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2?
=lstm_18/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_18/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_18/transpose:y:0Flstm_18/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_18/TensorArrayUnstack/TensorListFromTensor
lstm_18/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_18/strided_slice_2/stack
lstm_18/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_18/strided_slice_2/stack_1
lstm_18/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_18/strided_slice_2/stack_2Ќ
lstm_18/strided_slice_2StridedSlicelstm_18/transpose:y:0&lstm_18/strided_slice_2/stack:output:0(lstm_18/strided_slice_2/stack_1:output:0(lstm_18/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
lstm_18/strided_slice_2Э
*lstm_18/lstm_cell_18/MatMul/ReadVariableOpReadVariableOp3lstm_18_lstm_cell_18_matmul_readvariableop_resource*
_output_shapes
:	Ќ*
dtype02,
*lstm_18/lstm_cell_18/MatMul/ReadVariableOpЭ
lstm_18/lstm_cell_18/MatMulMatMul lstm_18/strided_slice_2:output:02lstm_18/lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_18/lstm_cell_18/MatMulг
,lstm_18/lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp5lstm_18_lstm_cell_18_matmul_1_readvariableop_resource*
_output_shapes
:	KЌ*
dtype02.
,lstm_18/lstm_cell_18/MatMul_1/ReadVariableOpЩ
lstm_18/lstm_cell_18/MatMul_1MatMullstm_18/zeros:output:04lstm_18/lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_18/lstm_cell_18/MatMul_1Р
lstm_18/lstm_cell_18/addAddV2%lstm_18/lstm_cell_18/MatMul:product:0'lstm_18/lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_18/lstm_cell_18/addЬ
+lstm_18/lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp4lstm_18_lstm_cell_18_biasadd_readvariableop_resource*
_output_shapes	
:Ќ*
dtype02-
+lstm_18/lstm_cell_18/BiasAdd/ReadVariableOpЭ
lstm_18/lstm_cell_18/BiasAddBiasAddlstm_18/lstm_cell_18/add:z:03lstm_18/lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_18/lstm_cell_18/BiasAddz
lstm_18/lstm_cell_18/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_18/lstm_cell_18/Const
$lstm_18/lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_18/lstm_cell_18/split/split_dim
lstm_18/lstm_cell_18/splitSplit-lstm_18/lstm_cell_18/split/split_dim:output:0%lstm_18/lstm_cell_18/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*
	num_split2
lstm_18/lstm_cell_18/split
lstm_18/lstm_cell_18/SigmoidSigmoid#lstm_18/lstm_cell_18/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_18/lstm_cell_18/SigmoidЂ
lstm_18/lstm_cell_18/Sigmoid_1Sigmoid#lstm_18/lstm_cell_18/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџK2 
lstm_18/lstm_cell_18/Sigmoid_1Ћ
lstm_18/lstm_cell_18/mulMul"lstm_18/lstm_cell_18/Sigmoid_1:y:0lstm_18/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_18/lstm_cell_18/mul
lstm_18/lstm_cell_18/ReluRelu#lstm_18/lstm_cell_18/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_18/lstm_cell_18/ReluМ
lstm_18/lstm_cell_18/mul_1Mul lstm_18/lstm_cell_18/Sigmoid:y:0'lstm_18/lstm_cell_18/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_18/lstm_cell_18/mul_1Б
lstm_18/lstm_cell_18/add_1AddV2lstm_18/lstm_cell_18/mul:z:0lstm_18/lstm_cell_18/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_18/lstm_cell_18/add_1Ђ
lstm_18/lstm_cell_18/Sigmoid_2Sigmoid#lstm_18/lstm_cell_18/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџK2 
lstm_18/lstm_cell_18/Sigmoid_2
lstm_18/lstm_cell_18/Relu_1Relulstm_18/lstm_cell_18/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_18/lstm_cell_18/Relu_1Р
lstm_18/lstm_cell_18/mul_2Mul"lstm_18/lstm_cell_18/Sigmoid_2:y:0)lstm_18/lstm_cell_18/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_18/lstm_cell_18/mul_2
%lstm_18/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   2'
%lstm_18/TensorArrayV2_1/element_shapeи
lstm_18/TensorArrayV2_1TensorListReserve.lstm_18/TensorArrayV2_1/element_shape:output:0 lstm_18/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_18/TensorArrayV2_1^
lstm_18/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_18/time
 lstm_18/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2"
 lstm_18/while/maximum_iterationsz
lstm_18/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_18/while/loop_counterъ
lstm_18/whileWhile#lstm_18/while/loop_counter:output:0)lstm_18/while/maximum_iterations:output:0lstm_18/time:output:0 lstm_18/TensorArrayV2_1:handle:0lstm_18/zeros:output:0lstm_18/zeros_1:output:0 lstm_18/strided_slice_1:output:0?lstm_18/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_18_lstm_cell_18_matmul_readvariableop_resource5lstm_18_lstm_cell_18_matmul_1_readvariableop_resource4lstm_18_lstm_cell_18_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџK:џџџџџџџџџK: : : : : *%
_read_only_resource_inputs
	
*'
bodyR
lstm_18_while_body_44729287*'
condR
lstm_18_while_cond_44729286*K
output_shapes:
8: : : : :џџџџџџџџџK:џџџџџџџџџK: : : : : *
parallel_iterations 2
lstm_18/whileХ
8lstm_18/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   2:
8lstm_18/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_18/TensorArrayV2Stack/TensorListStackTensorListStacklstm_18/while:output:3Alstm_18/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK*
element_dtype02,
*lstm_18/TensorArrayV2Stack/TensorListStack
lstm_18/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
lstm_18/strided_slice_3/stack
lstm_18/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_18/strided_slice_3/stack_1
lstm_18/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_18/strided_slice_3/stack_2Ъ
lstm_18/strided_slice_3StridedSlice3lstm_18/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_18/strided_slice_3/stack:output:0(lstm_18/strided_slice_3/stack_1:output:0(lstm_18/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџK*
shrink_axis_mask2
lstm_18/strided_slice_3
lstm_18/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_18/transpose_1/permЮ
lstm_18/transpose_1	Transpose3lstm_18/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_18/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
lstm_18/transpose_1v
lstm_18/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_18/runtimeP
gru_9/ShapeShapeinputs*
T0*
_output_shapes
:2
gru_9/Shape
gru_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_9/strided_slice/stack
gru_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_9/strided_slice/stack_1
gru_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_9/strided_slice/stack_2
gru_9/strided_sliceStridedSlicegru_9/Shape:output:0"gru_9/strided_slice/stack:output:0$gru_9/strided_slice/stack_1:output:0$gru_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_9/strided_sliceh
gru_9/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
gru_9/zeros/mul/y
gru_9/zeros/mulMulgru_9/strided_slice:output:0gru_9/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gru_9/zeros/mulk
gru_9/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
gru_9/zeros/Less/y
gru_9/zeros/LessLessgru_9/zeros/mul:z:0gru_9/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gru_9/zeros/Lessn
gru_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
gru_9/zeros/packed/1
gru_9/zeros/packedPackgru_9/strided_slice:output:0gru_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru_9/zeros/packedk
gru_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_9/zeros/Const
gru_9/zerosFillgru_9/zeros/packed:output:0gru_9/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_9/zeros
gru_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_9/transpose/perm
gru_9/transpose	Transposeinputsgru_9/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
gru_9/transposea
gru_9/Shape_1Shapegru_9/transpose:y:0*
T0*
_output_shapes
:2
gru_9/Shape_1
gru_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_9/strided_slice_1/stack
gru_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_9/strided_slice_1/stack_1
gru_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_9/strided_slice_1/stack_2
gru_9/strided_slice_1StridedSlicegru_9/Shape_1:output:0$gru_9/strided_slice_1/stack:output:0&gru_9/strided_slice_1/stack_1:output:0&gru_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_9/strided_slice_1
!gru_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!gru_9/TensorArrayV2/element_shapeЪ
gru_9/TensorArrayV2TensorListReserve*gru_9/TensorArrayV2/element_shape:output:0gru_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_9/TensorArrayV2Ы
;gru_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2=
;gru_9/TensorArrayUnstack/TensorListFromTensor/element_shape
-gru_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_9/transpose:y:0Dgru_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02/
-gru_9/TensorArrayUnstack/TensorListFromTensor
gru_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_9/strided_slice_2/stack
gru_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_9/strided_slice_2/stack_1
gru_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_9/strided_slice_2/stack_2 
gru_9/strided_slice_2StridedSlicegru_9/transpose:y:0$gru_9/strided_slice_2/stack:output:0&gru_9/strided_slice_2/stack_1:output:0&gru_9/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
gru_9/strided_slice_2Ќ
gru_9/gru_cell_9/ReadVariableOpReadVariableOp(gru_9_gru_cell_9_readvariableop_resource*
_output_shapes
:	*
dtype02!
gru_9/gru_cell_9/ReadVariableOp
gru_9/gru_cell_9/unstackUnpack'gru_9/gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
gru_9/gru_cell_9/unstackС
&gru_9/gru_cell_9/MatMul/ReadVariableOpReadVariableOp/gru_9_gru_cell_9_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02(
&gru_9/gru_cell_9/MatMul/ReadVariableOpП
gru_9/gru_cell_9/MatMulMatMulgru_9/strided_slice_2:output:0.gru_9/gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_9/gru_cell_9/MatMulИ
gru_9/gru_cell_9/BiasAddBiasAdd!gru_9/gru_cell_9/MatMul:product:0!gru_9/gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_9/gru_cell_9/BiasAddr
gru_9/gru_cell_9/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_9/gru_cell_9/Const
 gru_9/gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2"
 gru_9/gru_cell_9/split/split_dim№
gru_9/gru_cell_9/splitSplit)gru_9/gru_cell_9/split/split_dim:output:0!gru_9/gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
gru_9/gru_cell_9/splitЧ
(gru_9/gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp1gru_9_gru_cell_9_matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype02*
(gru_9/gru_cell_9/MatMul_1/ReadVariableOpЛ
gru_9/gru_cell_9/MatMul_1MatMulgru_9/zeros:output:00gru_9/gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_9/gru_cell_9/MatMul_1О
gru_9/gru_cell_9/BiasAdd_1BiasAdd#gru_9/gru_cell_9/MatMul_1:product:0!gru_9/gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_9/gru_cell_9/BiasAdd_1
gru_9/gru_cell_9/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   џџџџ2
gru_9/gru_cell_9/Const_1
"gru_9/gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2$
"gru_9/gru_cell_9/split_1/split_dimЈ
gru_9/gru_cell_9/split_1SplitV#gru_9/gru_cell_9/BiasAdd_1:output:0!gru_9/gru_cell_9/Const_1:output:0+gru_9/gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
gru_9/gru_cell_9/split_1Ћ
gru_9/gru_cell_9/addAddV2gru_9/gru_cell_9/split:output:0!gru_9/gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_9/gru_cell_9/add
gru_9/gru_cell_9/SigmoidSigmoidgru_9/gru_cell_9/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_9/gru_cell_9/SigmoidЏ
gru_9/gru_cell_9/add_1AddV2gru_9/gru_cell_9/split:output:1!gru_9/gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_9/gru_cell_9/add_1
gru_9/gru_cell_9/Sigmoid_1Sigmoidgru_9/gru_cell_9/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_9/gru_cell_9/Sigmoid_1Ј
gru_9/gru_cell_9/mulMulgru_9/gru_cell_9/Sigmoid_1:y:0!gru_9/gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_9/gru_cell_9/mulІ
gru_9/gru_cell_9/add_2AddV2gru_9/gru_cell_9/split:output:2gru_9/gru_cell_9/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_9/gru_cell_9/add_2
gru_9/gru_cell_9/ReluRelugru_9/gru_cell_9/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_9/gru_cell_9/Relu
gru_9/gru_cell_9/mul_1Mulgru_9/gru_cell_9/Sigmoid:y:0gru_9/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_9/gru_cell_9/mul_1u
gru_9/gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_9/gru_cell_9/sub/xЄ
gru_9/gru_cell_9/subSubgru_9/gru_cell_9/sub/x:output:0gru_9/gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_9/gru_cell_9/subЈ
gru_9/gru_cell_9/mul_2Mulgru_9/gru_cell_9/sub:z:0#gru_9/gru_cell_9/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_9/gru_cell_9/mul_2Ѓ
gru_9/gru_cell_9/add_3AddV2gru_9/gru_cell_9/mul_1:z:0gru_9/gru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_9/gru_cell_9/add_3
#gru_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2%
#gru_9/TensorArrayV2_1/element_shapeа
gru_9/TensorArrayV2_1TensorListReserve,gru_9/TensorArrayV2_1/element_shape:output:0gru_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_9/TensorArrayV2_1Z

gru_9/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

gru_9/time
gru_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2 
gru_9/while/maximum_iterationsv
gru_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_9/while/loop_counterџ
gru_9/whileWhile!gru_9/while/loop_counter:output:0'gru_9/while/maximum_iterations:output:0gru_9/time:output:0gru_9/TensorArrayV2_1:handle:0gru_9/zeros:output:0gru_9/strided_slice_1:output:0=gru_9/TensorArrayUnstack/TensorListFromTensor:output_handle:0(gru_9_gru_cell_9_readvariableop_resource/gru_9_gru_cell_9_matmul_readvariableop_resource1gru_9_gru_cell_9_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	*%
bodyR
gru_9_while_body_44729437*%
condR
gru_9_while_cond_44729436*8
output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *
parallel_iterations 2
gru_9/whileС
6gru_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   28
6gru_9/TensorArrayV2Stack/TensorListStack/element_shape
(gru_9/TensorArrayV2Stack/TensorListStackTensorListStackgru_9/while:output:3?gru_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02*
(gru_9/TensorArrayV2Stack/TensorListStack
gru_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
gru_9/strided_slice_3/stack
gru_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
gru_9/strided_slice_3/stack_1
gru_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_9/strided_slice_3/stack_2О
gru_9/strided_slice_3StridedSlice1gru_9/TensorArrayV2Stack/TensorListStack:tensor:0$gru_9/strided_slice_3/stack:output:0&gru_9/strided_slice_3/stack_1:output:0&gru_9/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
gru_9/strided_slice_3
gru_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_9/transpose_1/permЦ
gru_9/transpose_1	Transpose1gru_9/TensorArrayV2Stack/TensorListStack:tensor:0gru_9/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
gru_9/transpose_1r
gru_9/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_9/runtime
dropout_18/IdentityIdentitylstm_18/transpose_1:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
dropout_18/Identity
dropout_19/IdentityIdentitygru_9/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
dropout_19/Identityj
lstm_19/ShapeShapedropout_18/Identity:output:0*
T0*
_output_shapes
:2
lstm_19/Shape
lstm_19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_19/strided_slice/stack
lstm_19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_19/strided_slice/stack_1
lstm_19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_19/strided_slice/stack_2
lstm_19/strided_sliceStridedSlicelstm_19/Shape:output:0$lstm_19/strided_slice/stack:output:0&lstm_19/strided_slice/stack_1:output:0&lstm_19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_19/strided_slicel
lstm_19/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
lstm_19/zeros/mul/y
lstm_19/zeros/mulMullstm_19/strided_slice:output:0lstm_19/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_19/zeros/mulo
lstm_19/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_19/zeros/Less/y
lstm_19/zeros/LessLesslstm_19/zeros/mul:z:0lstm_19/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_19/zeros/Lessr
lstm_19/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
lstm_19/zeros/packed/1Ѓ
lstm_19/zeros/packedPacklstm_19/strided_slice:output:0lstm_19/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_19/zeros/packedo
lstm_19/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_19/zeros/Const
lstm_19/zerosFilllstm_19/zeros/packed:output:0lstm_19/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_19/zerosp
lstm_19/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
lstm_19/zeros_1/mul/y
lstm_19/zeros_1/mulMullstm_19/strided_slice:output:0lstm_19/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_19/zeros_1/muls
lstm_19/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_19/zeros_1/Less/y
lstm_19/zeros_1/LessLesslstm_19/zeros_1/mul:z:0lstm_19/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_19/zeros_1/Lessv
lstm_19/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
lstm_19/zeros_1/packed/1Љ
lstm_19/zeros_1/packedPacklstm_19/strided_slice:output:0!lstm_19/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_19/zeros_1/packeds
lstm_19/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_19/zeros_1/Const
lstm_19/zeros_1Filllstm_19/zeros_1/packed:output:0lstm_19/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_19/zeros_1
lstm_19/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_19/transpose/permБ
lstm_19/transpose	Transposedropout_18/Identity:output:0lstm_19/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
lstm_19/transposeg
lstm_19/Shape_1Shapelstm_19/transpose:y:0*
T0*
_output_shapes
:2
lstm_19/Shape_1
lstm_19/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_19/strided_slice_1/stack
lstm_19/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_19/strided_slice_1/stack_1
lstm_19/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_19/strided_slice_1/stack_2
lstm_19/strided_slice_1StridedSlicelstm_19/Shape_1:output:0&lstm_19/strided_slice_1/stack:output:0(lstm_19/strided_slice_1/stack_1:output:0(lstm_19/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_19/strided_slice_1
#lstm_19/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#lstm_19/TensorArrayV2/element_shapeв
lstm_19/TensorArrayV2TensorListReserve,lstm_19/TensorArrayV2/element_shape:output:0 lstm_19/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_19/TensorArrayV2Я
=lstm_19/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   2?
=lstm_19/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_19/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_19/transpose:y:0Flstm_19/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_19/TensorArrayUnstack/TensorListFromTensor
lstm_19/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_19/strided_slice_2/stack
lstm_19/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_19/strided_slice_2/stack_1
lstm_19/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_19/strided_slice_2/stack_2Ќ
lstm_19/strided_slice_2StridedSlicelstm_19/transpose:y:0&lstm_19/strided_slice_2/stack:output:0(lstm_19/strided_slice_2/stack_1:output:0(lstm_19/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџK*
shrink_axis_mask2
lstm_19/strided_slice_2Э
*lstm_19/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp3lstm_19_lstm_cell_19_matmul_readvariableop_resource*
_output_shapes
:	KШ*
dtype02,
*lstm_19/lstm_cell_19/MatMul/ReadVariableOpЭ
lstm_19/lstm_cell_19/MatMulMatMul lstm_19/strided_slice_2:output:02lstm_19/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_19/lstm_cell_19/MatMulг
,lstm_19/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp5lstm_19_lstm_cell_19_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02.
,lstm_19/lstm_cell_19/MatMul_1/ReadVariableOpЩ
lstm_19/lstm_cell_19/MatMul_1MatMullstm_19/zeros:output:04lstm_19/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_19/lstm_cell_19/MatMul_1Р
lstm_19/lstm_cell_19/addAddV2%lstm_19/lstm_cell_19/MatMul:product:0'lstm_19/lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_19/lstm_cell_19/addЬ
+lstm_19/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp4lstm_19_lstm_cell_19_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02-
+lstm_19/lstm_cell_19/BiasAdd/ReadVariableOpЭ
lstm_19/lstm_cell_19/BiasAddBiasAddlstm_19/lstm_cell_19/add:z:03lstm_19/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_19/lstm_cell_19/BiasAddz
lstm_19/lstm_cell_19/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_19/lstm_cell_19/Const
$lstm_19/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_19/lstm_cell_19/split/split_dim
lstm_19/lstm_cell_19/splitSplit-lstm_19/lstm_cell_19/split/split_dim:output:0%lstm_19/lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_19/lstm_cell_19/split
lstm_19/lstm_cell_19/SigmoidSigmoid#lstm_19/lstm_cell_19/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_19/lstm_cell_19/SigmoidЂ
lstm_19/lstm_cell_19/Sigmoid_1Sigmoid#lstm_19/lstm_cell_19/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22 
lstm_19/lstm_cell_19/Sigmoid_1Ћ
lstm_19/lstm_cell_19/mulMul"lstm_19/lstm_cell_19/Sigmoid_1:y:0lstm_19/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_19/lstm_cell_19/mul
lstm_19/lstm_cell_19/ReluRelu#lstm_19/lstm_cell_19/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_19/lstm_cell_19/ReluМ
lstm_19/lstm_cell_19/mul_1Mul lstm_19/lstm_cell_19/Sigmoid:y:0'lstm_19/lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_19/lstm_cell_19/mul_1Б
lstm_19/lstm_cell_19/add_1AddV2lstm_19/lstm_cell_19/mul:z:0lstm_19/lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_19/lstm_cell_19/add_1Ђ
lstm_19/lstm_cell_19/Sigmoid_2Sigmoid#lstm_19/lstm_cell_19/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22 
lstm_19/lstm_cell_19/Sigmoid_2
lstm_19/lstm_cell_19/Relu_1Relulstm_19/lstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_19/lstm_cell_19/Relu_1Р
lstm_19/lstm_cell_19/mul_2Mul"lstm_19/lstm_cell_19/Sigmoid_2:y:0)lstm_19/lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_19/lstm_cell_19/mul_2
%lstm_19/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2'
%lstm_19/TensorArrayV2_1/element_shapeи
lstm_19/TensorArrayV2_1TensorListReserve.lstm_19/TensorArrayV2_1/element_shape:output:0 lstm_19/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_19/TensorArrayV2_1^
lstm_19/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_19/time
 lstm_19/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2"
 lstm_19/while/maximum_iterationsz
lstm_19/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_19/while/loop_counterъ
lstm_19/whileWhile#lstm_19/while/loop_counter:output:0)lstm_19/while/maximum_iterations:output:0lstm_19/time:output:0 lstm_19/TensorArrayV2_1:handle:0lstm_19/zeros:output:0lstm_19/zeros_1:output:0 lstm_19/strided_slice_1:output:0?lstm_19/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_19_lstm_cell_19_matmul_readvariableop_resource5lstm_19_lstm_cell_19_matmul_1_readvariableop_resource4lstm_19_lstm_cell_19_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*'
bodyR
lstm_19_while_body_44729593*'
condR
lstm_19_while_cond_44729592*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
lstm_19/whileХ
8lstm_19/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2:
8lstm_19/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_19/TensorArrayV2Stack/TensorListStackTensorListStacklstm_19/while:output:3Alstm_19/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02,
*lstm_19/TensorArrayV2Stack/TensorListStack
lstm_19/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
lstm_19/strided_slice_3/stack
lstm_19/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_19/strided_slice_3/stack_1
lstm_19/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_19/strided_slice_3/stack_2Ъ
lstm_19/strided_slice_3StridedSlice3lstm_19/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_19/strided_slice_3/stack:output:0(lstm_19/strided_slice_3/stack_1:output:0(lstm_19/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
lstm_19/strided_slice_3
lstm_19/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_19/transpose_1/permЮ
lstm_19/transpose_1	Transpose3lstm_19/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_19/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
lstm_19/transpose_1v
lstm_19/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_19/runtimeЈ
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*
_output_shapes

:2 *
dtype02 
dense_27/MatMul/ReadVariableOpЈ
dense_27/MatMulMatMul lstm_19/strided_slice_3:output:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_27/MatMulЇ
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_27/BiasAdd/ReadVariableOpЅ
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_27/BiasAdds
dense_27/ReluReludense_27/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_27/ReluЈ
dense_28/MatMul/ReadVariableOpReadVariableOp'dense_28_matmul_readvariableop_resource*
_output_shapes

:2@*
dtype02 
dense_28/MatMul/ReadVariableOpЄ
dense_28/MatMulMatMuldropout_19/Identity:output:0&dense_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_28/MatMulЇ
dense_28/BiasAdd/ReadVariableOpReadVariableOp(dense_28_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_28/BiasAdd/ReadVariableOpЅ
dense_28/BiasAddBiasAdddense_28/MatMul:product:0'dense_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_28/BiasAdds
dense_28/ReluReludense_28/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_28/Relux
concatenate_9/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_9/concat/axisб
concatenate_9/concatConcatV2dense_27/Relu:activations:0dense_28/Relu:activations:0"concatenate_9/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ`2
concatenate_9/concatЈ
dense_29/MatMul/ReadVariableOpReadVariableOp'dense_29_matmul_readvariableop_resource*
_output_shapes

:`*
dtype02 
dense_29/MatMul/ReadVariableOpЅ
dense_29/MatMulMatMulconcatenate_9/concat:output:0&dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_29/MatMulЇ
dense_29/BiasAdd/ReadVariableOpReadVariableOp(dense_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_29/BiasAdd/ReadVariableOpЅ
dense_29/BiasAddBiasAdddense_29/MatMul:product:0'dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_29/BiasAddю
IdentityIdentitydense_29/BiasAdd:output:0 ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp ^dense_28/BiasAdd/ReadVariableOp^dense_28/MatMul/ReadVariableOp ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOp'^gru_9/gru_cell_9/MatMul/ReadVariableOp)^gru_9/gru_cell_9/MatMul_1/ReadVariableOp ^gru_9/gru_cell_9/ReadVariableOp^gru_9/while,^lstm_18/lstm_cell_18/BiasAdd/ReadVariableOp+^lstm_18/lstm_cell_18/MatMul/ReadVariableOp-^lstm_18/lstm_cell_18/MatMul_1/ReadVariableOp^lstm_18/while,^lstm_19/lstm_cell_19/BiasAdd/ReadVariableOp+^lstm_19/lstm_cell_19/MatMul/ReadVariableOp-^lstm_19/lstm_cell_19/MatMul_1/ReadVariableOp^lstm_19/while*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:џџџџџџџџџџџџџџџџџџ:::::::::::::::2B
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp2B
dense_28/BiasAdd/ReadVariableOpdense_28/BiasAdd/ReadVariableOp2@
dense_28/MatMul/ReadVariableOpdense_28/MatMul/ReadVariableOp2B
dense_29/BiasAdd/ReadVariableOpdense_29/BiasAdd/ReadVariableOp2@
dense_29/MatMul/ReadVariableOpdense_29/MatMul/ReadVariableOp2P
&gru_9/gru_cell_9/MatMul/ReadVariableOp&gru_9/gru_cell_9/MatMul/ReadVariableOp2T
(gru_9/gru_cell_9/MatMul_1/ReadVariableOp(gru_9/gru_cell_9/MatMul_1/ReadVariableOp2B
gru_9/gru_cell_9/ReadVariableOpgru_9/gru_cell_9/ReadVariableOp2
gru_9/whilegru_9/while2Z
+lstm_18/lstm_cell_18/BiasAdd/ReadVariableOp+lstm_18/lstm_cell_18/BiasAdd/ReadVariableOp2X
*lstm_18/lstm_cell_18/MatMul/ReadVariableOp*lstm_18/lstm_cell_18/MatMul/ReadVariableOp2\
,lstm_18/lstm_cell_18/MatMul_1/ReadVariableOp,lstm_18/lstm_cell_18/MatMul_1/ReadVariableOp2
lstm_18/whilelstm_18/while2Z
+lstm_19/lstm_cell_19/BiasAdd/ReadVariableOp+lstm_19/lstm_cell_19/BiasAdd/ReadVariableOp2X
*lstm_19/lstm_cell_19/MatMul/ReadVariableOp*lstm_19/lstm_cell_19/MatMul/ReadVariableOp2\
,lstm_19/lstm_cell_19/MatMul_1/ReadVariableOp,lstm_19/lstm_cell_19/MatMul_1/ReadVariableOp2
lstm_19/whilelstm_19/while:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
C

while_body_44728114
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_19_matmul_readvariableop_resource_09
5while_lstm_cell_19_matmul_1_readvariableop_resource_08
4while_lstm_cell_19_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_19_matmul_readvariableop_resource7
3while_lstm_cell_19_matmul_1_readvariableop_resource6
2while_lstm_cell_19_biasadd_readvariableop_resourceЂ)while/lstm_cell_19/BiasAdd/ReadVariableOpЂ(while/lstm_cell_19/MatMul/ReadVariableOpЂ*while/lstm_cell_19/MatMul_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџK*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЩ
(while/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_19_matmul_readvariableop_resource_0*
_output_shapes
:	KШ*
dtype02*
(while/lstm_cell_19/MatMul/ReadVariableOpз
while/lstm_cell_19/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_19/MatMulЯ
*while/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_19_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02,
*while/lstm_cell_19/MatMul_1/ReadVariableOpР
while/lstm_cell_19/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_19/MatMul_1И
while/lstm_cell_19/addAddV2#while/lstm_cell_19/MatMul:product:0%while/lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_19/addШ
)while/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_19_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02+
)while/lstm_cell_19/BiasAdd/ReadVariableOpХ
while/lstm_cell_19/BiasAddBiasAddwhile/lstm_cell_19/add:z:01while/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_19/BiasAddv
while/lstm_cell_19/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_19/Const
"while/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_19/split/split_dim
while/lstm_cell_19/splitSplit+while/lstm_cell_19/split/split_dim:output:0#while/lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_19/split
while/lstm_cell_19/SigmoidSigmoid!while/lstm_cell_19/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/Sigmoid
while/lstm_cell_19/Sigmoid_1Sigmoid!while/lstm_cell_19/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/Sigmoid_1 
while/lstm_cell_19/mulMul while/lstm_cell_19/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/mul
while/lstm_cell_19/ReluRelu!while/lstm_cell_19/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/ReluД
while/lstm_cell_19/mul_1Mulwhile/lstm_cell_19/Sigmoid:y:0%while/lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/mul_1Љ
while/lstm_cell_19/add_1AddV2while/lstm_cell_19/mul:z:0while/lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/add_1
while/lstm_cell_19/Sigmoid_2Sigmoid!while/lstm_cell_19/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/Sigmoid_2
while/lstm_cell_19/Relu_1Reluwhile/lstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/Relu_1И
while/lstm_cell_19/mul_2Mul while/lstm_cell_19/Sigmoid_2:y:0'while/lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_19/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1т
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityѕ
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1ф
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_19/mul_2:z:0*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_19/add_1:z:0*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_19_biasadd_readvariableop_resource4while_lstm_cell_19_biasadd_readvariableop_resource_0"l
3while_lstm_cell_19_matmul_1_readvariableop_resource5while_lstm_cell_19_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_19_matmul_readvariableop_resource3while_lstm_cell_19_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : :::2V
)while/lstm_cell_19/BiasAdd/ReadVariableOp)while/lstm_cell_19/BiasAdd/ReadVariableOp2T
(while/lstm_cell_19/MatMul/ReadVariableOp(while/lstm_cell_19/MatMul/ReadVariableOp2X
*while/lstm_cell_19/MatMul_1/ReadVariableOp*while/lstm_cell_19/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
Х[
є
E__inference_lstm_19_layer_call_and_return_conditional_losses_44731614

inputs/
+lstm_cell_19_matmul_readvariableop_resource1
-lstm_cell_19_matmul_1_readvariableop_resource0
,lstm_cell_19_biasadd_readvariableop_resource
identityЂ#lstm_cell_19/BiasAdd/ReadVariableOpЂ"lstm_cell_19/MatMul/ReadVariableOpЂ$lstm_cell_19/MatMul_1/ReadVariableOpЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџK*
shrink_axis_mask2
strided_slice_2Е
"lstm_cell_19/MatMul/ReadVariableOpReadVariableOp+lstm_cell_19_matmul_readvariableop_resource*
_output_shapes
:	KШ*
dtype02$
"lstm_cell_19/MatMul/ReadVariableOp­
lstm_cell_19/MatMulMatMulstrided_slice_2:output:0*lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_19/MatMulЛ
$lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_19_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02&
$lstm_cell_19/MatMul_1/ReadVariableOpЉ
lstm_cell_19/MatMul_1MatMulzeros:output:0,lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_19/MatMul_1 
lstm_cell_19/addAddV2lstm_cell_19/MatMul:product:0lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_19/addД
#lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_19_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02%
#lstm_cell_19/BiasAdd/ReadVariableOp­
lstm_cell_19/BiasAddBiasAddlstm_cell_19/add:z:0+lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_19/BiasAddj
lstm_cell_19/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_19/Const~
lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_19/split/split_dimѓ
lstm_cell_19/splitSplit%lstm_cell_19/split/split_dim:output:0lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_19/split
lstm_cell_19/SigmoidSigmoidlstm_cell_19/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/Sigmoid
lstm_cell_19/Sigmoid_1Sigmoidlstm_cell_19/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/Sigmoid_1
lstm_cell_19/mulMullstm_cell_19/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/mul}
lstm_cell_19/ReluRelulstm_cell_19/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/Relu
lstm_cell_19/mul_1Mullstm_cell_19/Sigmoid:y:0lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/mul_1
lstm_cell_19/add_1AddV2lstm_cell_19/mul:z:0lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/add_1
lstm_cell_19/Sigmoid_2Sigmoidlstm_cell_19/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/Sigmoid_2|
lstm_cell_19/Relu_1Relulstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/Relu_1 
lstm_cell_19/mul_2Mullstm_cell_19/Sigmoid_2:y:0!lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterђ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_19_matmul_readvariableop_resource-lstm_cell_19_matmul_1_readvariableop_resource,lstm_cell_19_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_44731529*
condR
while_cond_44731528*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeц
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_19/BiasAdd/ReadVariableOp#^lstm_cell_19/MatMul/ReadVariableOp%^lstm_cell_19/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџK:::2J
#lstm_cell_19/BiasAdd/ReadVariableOp#lstm_cell_19/BiasAdd/ReadVariableOp2H
"lstm_cell_19/MatMul/ReadVariableOp"lstm_cell_19/MatMul/ReadVariableOp2L
$lstm_cell_19/MatMul_1/ReadVariableOp$lstm_cell_19/MatMul_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK
 
_user_specified_nameinputs
щZ
ж
C__inference_gru_9_layer_call_and_return_conditional_losses_44727957

inputs&
"gru_cell_9_readvariableop_resource-
)gru_cell_9_matmul_readvariableop_resource/
+gru_cell_9_matmul_1_readvariableop_resource
identityЂ gru_cell_9/MatMul/ReadVariableOpЂ"gru_cell_9/MatMul_1/ReadVariableOpЂgru_cell_9/ReadVariableOpЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2
gru_cell_9/ReadVariableOpReadVariableOp"gru_cell_9_readvariableop_resource*
_output_shapes
:	*
dtype02
gru_cell_9/ReadVariableOp
gru_cell_9/unstackUnpack!gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
gru_cell_9/unstackЏ
 gru_cell_9/MatMul/ReadVariableOpReadVariableOp)gru_cell_9_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 gru_cell_9/MatMul/ReadVariableOpЇ
gru_cell_9/MatMulMatMulstrided_slice_2:output:0(gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_cell_9/MatMul 
gru_cell_9/BiasAddBiasAddgru_cell_9/MatMul:product:0gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_cell_9/BiasAddf
gru_cell_9/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_9/Const
gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_9/split/split_dimи
gru_cell_9/splitSplit#gru_cell_9/split/split_dim:output:0gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
gru_cell_9/splitЕ
"gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_9_matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype02$
"gru_cell_9/MatMul_1/ReadVariableOpЃ
gru_cell_9/MatMul_1MatMulzeros:output:0*gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_cell_9/MatMul_1І
gru_cell_9/BiasAdd_1BiasAddgru_cell_9/MatMul_1:product:0gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_cell_9/BiasAdd_1}
gru_cell_9/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   џџџџ2
gru_cell_9/Const_1
gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_9/split_1/split_dim
gru_cell_9/split_1SplitVgru_cell_9/BiasAdd_1:output:0gru_cell_9/Const_1:output:0%gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
gru_cell_9/split_1
gru_cell_9/addAddV2gru_cell_9/split:output:0gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/addy
gru_cell_9/SigmoidSigmoidgru_cell_9/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/Sigmoid
gru_cell_9/add_1AddV2gru_cell_9/split:output:1gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/add_1
gru_cell_9/Sigmoid_1Sigmoidgru_cell_9/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/Sigmoid_1
gru_cell_9/mulMulgru_cell_9/Sigmoid_1:y:0gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/mul
gru_cell_9/add_2AddV2gru_cell_9/split:output:2gru_cell_9/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/add_2r
gru_cell_9/ReluRelugru_cell_9/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/Relu
gru_cell_9/mul_1Mulgru_cell_9/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/mul_1i
gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_9/sub/x
gru_cell_9/subSubgru_cell_9/sub/x:output:0gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/sub
gru_cell_9/mul_2Mulgru_cell_9/sub:z:0gru_cell_9/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/mul_2
gru_cell_9/add_3AddV2gru_cell_9/mul_1:z:0gru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЋ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_9_readvariableop_resource)gru_cell_9_matmul_readvariableop_resource+gru_cell_9_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_44727867*
condR
while_cond_44727866*8
output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeи
IdentityIdentitystrided_slice_3:output:0!^gru_cell_9/MatMul/ReadVariableOp#^gru_cell_9/MatMul_1/ReadVariableOp^gru_cell_9/ReadVariableOp^while*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2D
 gru_cell_9/MatMul/ReadVariableOp gru_cell_9/MatMul/ReadVariableOp2H
"gru_cell_9/MatMul_1/ReadVariableOp"gru_cell_9/MatMul_1/ReadVariableOp26
gru_cell_9/ReadVariableOpgru_cell_9/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
	
І
gru_9_while_cond_44729436(
$gru_9_while_gru_9_while_loop_counter.
*gru_9_while_gru_9_while_maximum_iterations
gru_9_while_placeholder
gru_9_while_placeholder_1
gru_9_while_placeholder_2*
&gru_9_while_less_gru_9_strided_slice_1B
>gru_9_while_gru_9_while_cond_44729436___redundant_placeholder0B
>gru_9_while_gru_9_while_cond_44729436___redundant_placeholder1B
>gru_9_while_gru_9_while_cond_44729436___redundant_placeholder2B
>gru_9_while_gru_9_while_cond_44729436___redundant_placeholder3
gru_9_while_identity

gru_9/while/LessLessgru_9_while_placeholder&gru_9_while_less_gru_9_strided_slice_1*
T0*
_output_shapes
: 2
gru_9/while/Lesso
gru_9/while/IdentityIdentitygru_9/while/Less:z:0*
T0
*
_output_shapes
: 2
gru_9/while/Identity"5
gru_9_while_identitygru_9/while/Identity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:


(__inference_gru_9_layer_call_fn_44731122
inputs_0
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_gru_9_layer_call_and_return_conditional_losses_447265642
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
Т
Я
/__inference_lstm_cell_18_layer_call_fn_44731971

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2ЂStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_lstm_cell_18_layer_call_and_return_conditional_losses_447255922
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџK2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџK2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџK2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:џџџџџџџџџ:џџџџџџџџџK:џџџџџџџџџK:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџK
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџK
"
_user_specified_name
states/1
Е
Э
while_cond_44731681
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_44731681___redundant_placeholder06
2while_while_cond_44731681___redundant_placeholder16
2while_while_cond_44731681___redundant_placeholder26
2while_while_cond_44731681___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџ2:џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
Р
w
K__inference_concatenate_9_layer_call_and_return_conditional_losses_44731863
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ`2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:џџџџџџџџџ :џџџџџџџџџ@:Q M
'
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:џџџџџџџџџ@
"
_user_specified_name
inputs/1
Х[
є
E__inference_lstm_19_layer_call_and_return_conditional_losses_44728352

inputs/
+lstm_cell_19_matmul_readvariableop_resource1
-lstm_cell_19_matmul_1_readvariableop_resource0
,lstm_cell_19_biasadd_readvariableop_resource
identityЂ#lstm_cell_19/BiasAdd/ReadVariableOpЂ"lstm_cell_19/MatMul/ReadVariableOpЂ$lstm_cell_19/MatMul_1/ReadVariableOpЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџK*
shrink_axis_mask2
strided_slice_2Е
"lstm_cell_19/MatMul/ReadVariableOpReadVariableOp+lstm_cell_19_matmul_readvariableop_resource*
_output_shapes
:	KШ*
dtype02$
"lstm_cell_19/MatMul/ReadVariableOp­
lstm_cell_19/MatMulMatMulstrided_slice_2:output:0*lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_19/MatMulЛ
$lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_19_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02&
$lstm_cell_19/MatMul_1/ReadVariableOpЉ
lstm_cell_19/MatMul_1MatMulzeros:output:0,lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_19/MatMul_1 
lstm_cell_19/addAddV2lstm_cell_19/MatMul:product:0lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_19/addД
#lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_19_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02%
#lstm_cell_19/BiasAdd/ReadVariableOp­
lstm_cell_19/BiasAddBiasAddlstm_cell_19/add:z:0+lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_19/BiasAddj
lstm_cell_19/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_19/Const~
lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_19/split/split_dimѓ
lstm_cell_19/splitSplit%lstm_cell_19/split/split_dim:output:0lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_19/split
lstm_cell_19/SigmoidSigmoidlstm_cell_19/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/Sigmoid
lstm_cell_19/Sigmoid_1Sigmoidlstm_cell_19/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/Sigmoid_1
lstm_cell_19/mulMullstm_cell_19/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/mul}
lstm_cell_19/ReluRelulstm_cell_19/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/Relu
lstm_cell_19/mul_1Mullstm_cell_19/Sigmoid:y:0lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/mul_1
lstm_cell_19/add_1AddV2lstm_cell_19/mul:z:0lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/add_1
lstm_cell_19/Sigmoid_2Sigmoidlstm_cell_19/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/Sigmoid_2|
lstm_cell_19/Relu_1Relulstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/Relu_1 
lstm_cell_19/mul_2Mullstm_cell_19/Sigmoid_2:y:0!lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterђ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_19_matmul_readvariableop_resource-lstm_cell_19_matmul_1_readvariableop_resource,lstm_cell_19_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_44728267*
condR
while_cond_44728266*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeц
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_19/BiasAdd/ReadVariableOp#^lstm_cell_19/MatMul/ReadVariableOp%^lstm_cell_19/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџK:::2J
#lstm_cell_19/BiasAdd/ReadVariableOp#lstm_cell_19/BiasAdd/ReadVariableOp2H
"lstm_cell_19/MatMul/ReadVariableOp"lstm_cell_19/MatMul/ReadVariableOp2L
$lstm_cell_19/MatMul_1/ReadVariableOp$lstm_cell_19/MatMul_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK
 
_user_specified_nameinputs
ф

+__inference_dense_29_layer_call_fn_44731888

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_29_layer_call_and_return_conditional_losses_447284622
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ`::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ`
 
_user_specified_nameinputs
Щ[
є
E__inference_lstm_18_layer_call_and_return_conditional_losses_44727610

inputs/
+lstm_cell_18_matmul_readvariableop_resource1
-lstm_cell_18_matmul_1_readvariableop_resource0
,lstm_cell_18_biasadd_readvariableop_resource
identityЂ#lstm_cell_18/BiasAdd/ReadVariableOpЂ"lstm_cell_18/MatMul/ReadVariableOpЂ$lstm_cell_18/MatMul_1/ReadVariableOpЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2Е
"lstm_cell_18/MatMul/ReadVariableOpReadVariableOp+lstm_cell_18_matmul_readvariableop_resource*
_output_shapes
:	Ќ*
dtype02$
"lstm_cell_18/MatMul/ReadVariableOp­
lstm_cell_18/MatMulMatMulstrided_slice_2:output:0*lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_18/MatMulЛ
$lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_18_matmul_1_readvariableop_resource*
_output_shapes
:	KЌ*
dtype02&
$lstm_cell_18/MatMul_1/ReadVariableOpЉ
lstm_cell_18/MatMul_1MatMulzeros:output:0,lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_18/MatMul_1 
lstm_cell_18/addAddV2lstm_cell_18/MatMul:product:0lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_18/addД
#lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_18_biasadd_readvariableop_resource*
_output_shapes	
:Ќ*
dtype02%
#lstm_cell_18/BiasAdd/ReadVariableOp­
lstm_cell_18/BiasAddBiasAddlstm_cell_18/add:z:0+lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_18/BiasAddj
lstm_cell_18/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_18/Const~
lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_18/split/split_dimѓ
lstm_cell_18/splitSplit%lstm_cell_18/split/split_dim:output:0lstm_cell_18/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*
	num_split2
lstm_cell_18/split
lstm_cell_18/SigmoidSigmoidlstm_cell_18/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/Sigmoid
lstm_cell_18/Sigmoid_1Sigmoidlstm_cell_18/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/Sigmoid_1
lstm_cell_18/mulMullstm_cell_18/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/mul}
lstm_cell_18/ReluRelulstm_cell_18/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/Relu
lstm_cell_18/mul_1Mullstm_cell_18/Sigmoid:y:0lstm_cell_18/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/mul_1
lstm_cell_18/add_1AddV2lstm_cell_18/mul:z:0lstm_cell_18/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/add_1
lstm_cell_18/Sigmoid_2Sigmoidlstm_cell_18/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/Sigmoid_2|
lstm_cell_18/Relu_1Relulstm_cell_18/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/Relu_1 
lstm_cell_18/mul_2Mullstm_cell_18/Sigmoid_2:y:0!lstm_cell_18/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterђ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_18_matmul_readvariableop_resource-lstm_cell_18_matmul_1_readvariableop_resource,lstm_cell_18_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџK:џџџџџџџџџK: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_44727525*
condR
while_cond_44727524*K
output_shapes:
8: : : : :џџџџџџџџџK:џџџџџџџџџK: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџK*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeъ
IdentityIdentitytranspose_1:y:0$^lstm_cell_18/BiasAdd/ReadVariableOp#^lstm_cell_18/MatMul/ReadVariableOp%^lstm_cell_18/MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2J
#lstm_cell_18/BiasAdd/ReadVariableOp#lstm_cell_18/BiasAdd/ReadVariableOp2H
"lstm_cell_18/MatMul/ReadVariableOp"lstm_cell_18/MatMul/ReadVariableOp2L
$lstm_cell_18/MatMul_1/ReadVariableOp$lstm_cell_18/MatMul_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
­
н
J__inference_lstm_cell_18_layer_call_and_return_conditional_losses_44725625

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ќ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	KЌ*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ќ*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimП
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџK2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџK2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџK2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџK2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
mul_2Ј
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2

IdentityЌ

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2

Identity_1Ќ

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:џџџџџџџџџ:џџџџџџџџџK:џџџџџџџџџK:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџK
 
_user_specified_namestates:OK
'
_output_shapes
:џџџџџџџџџK
 
_user_specified_namestates
­
н
J__inference_lstm_cell_18_layer_call_and_return_conditional_losses_44725592

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ќ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	KЌ*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ќ*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimП
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџK2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџK2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџK2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџK2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
mul_2Ј
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2

IdentityЌ

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2

Identity_1Ќ

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:џџџџџџџџџ:џџџџџџџџџK:џџџџџџџџџK:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџK
 
_user_specified_namestates:OK
'
_output_shapes
:џџџџџџџџџK
 
_user_specified_namestates
Р%

while_body_44727223
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0!
while_lstm_cell_19_44727247_0!
while_lstm_cell_19_44727249_0!
while_lstm_cell_19_44727251_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_19_44727247
while_lstm_cell_19_44727249
while_lstm_cell_19_44727251Ђ*while/lstm_cell_19/StatefulPartitionedCallУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџK*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemь
*while/lstm_cell_19/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_19_44727247_0while_lstm_cell_19_44727249_0while_lstm_cell_19_44727251_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_lstm_cell_19_layer_call_and_return_conditional_losses_447267972,
*while/lstm_cell_19/StatefulPartitionedCallї
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_19/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0+^while/lstm_cell_19/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations+^while/lstm_cell_19/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0+^while/lstm_cell_19/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2К
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0+^while/lstm_cell_19/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Ф
while/Identity_4Identity3while/lstm_cell_19/StatefulPartitionedCall:output:1+^while/lstm_cell_19/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4Ф
while/Identity_5Identity3while/lstm_cell_19/StatefulPartitionedCall:output:2+^while/lstm_cell_19/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"<
while_lstm_cell_19_44727247while_lstm_cell_19_44727247_0"<
while_lstm_cell_19_44727249while_lstm_cell_19_44727249_0"<
while_lstm_cell_19_44727251while_lstm_cell_19_44727251_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : :::2X
*while/lstm_cell_19/StatefulPartitionedCall*while/lstm_cell_19/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
Р%

while_body_44727091
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0!
while_lstm_cell_19_44727115_0!
while_lstm_cell_19_44727117_0!
while_lstm_cell_19_44727119_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_19_44727115
while_lstm_cell_19_44727117
while_lstm_cell_19_44727119Ђ*while/lstm_cell_19/StatefulPartitionedCallУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџK*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemь
*while/lstm_cell_19/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_19_44727115_0while_lstm_cell_19_44727117_0while_lstm_cell_19_44727119_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_lstm_cell_19_layer_call_and_return_conditional_losses_447267642,
*while/lstm_cell_19/StatefulPartitionedCallї
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_19/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0+^while/lstm_cell_19/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations+^while/lstm_cell_19/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0+^while/lstm_cell_19/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2К
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0+^while/lstm_cell_19/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Ф
while/Identity_4Identity3while/lstm_cell_19/StatefulPartitionedCall:output:1+^while/lstm_cell_19/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4Ф
while/Identity_5Identity3while/lstm_cell_19/StatefulPartitionedCall:output:2+^while/lstm_cell_19/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"<
while_lstm_cell_19_44727115while_lstm_cell_19_44727115_0"<
while_lstm_cell_19_44727117while_lstm_cell_19_44727117_0"<
while_lstm_cell_19_44727119while_lstm_cell_19_44727119_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : :::2X
*while/lstm_cell_19/StatefulPartitionedCall*while/lstm_cell_19/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
Е
Э
while_cond_44728113
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_44728113___redundant_placeholder06
2while_while_cond_44728113___redundant_placeholder16
2while_while_cond_44728113___redundant_placeholder26
2while_while_cond_44728113___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџ2:џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
Е
п
J__inference_lstm_cell_19_layer_call_and_return_conditional_losses_44732129

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	KШ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimП
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџ22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
mul_2Ј
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22

IdentityЌ

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_1Ќ

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:џџџџџџџџџK:џџџџџџџџџ2:џџџџџџџџџ2:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџK
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ2
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџ2
"
_user_specified_name
states/1
Т
Я
/__inference_lstm_cell_18_layer_call_fn_44731988

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2ЂStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_lstm_cell_18_layer_call_and_return_conditional_losses_447256252
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџK2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџK2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџK2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:џџџџџџџџџ:џџџџџџџџџK:џџџџџџџџџK:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџK
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџK
"
_user_specified_name
states/1
џ
f
H__inference_dropout_18_layer_call_and_return_conditional_losses_44728004

inputs

identity_1g
IdentityIdentityinputs*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2

Identityv

Identity_1IdentityIdentity:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2

Identity_1"!

identity_1Identity_1:output:0*3
_input_shapes"
 :џџџџџџџџџџџџџџџџџџK:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK
 
_user_specified_nameinputs
Е
Э
while_cond_44731528
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_44731528___redundant_placeholder06
2while_while_cond_44731528___redundant_placeholder16
2while_while_cond_44731528___redundant_placeholder26
2while_while_cond_44731528___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџ2:џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
C

while_body_44727372
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_18_matmul_readvariableop_resource_09
5while_lstm_cell_18_matmul_1_readvariableop_resource_08
4while_lstm_cell_18_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_18_matmul_readvariableop_resource7
3while_lstm_cell_18_matmul_1_readvariableop_resource6
2while_lstm_cell_18_biasadd_readvariableop_resourceЂ)while/lstm_cell_18/BiasAdd/ReadVariableOpЂ(while/lstm_cell_18/MatMul/ReadVariableOpЂ*while/lstm_cell_18/MatMul_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЩ
(while/lstm_cell_18/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_18_matmul_readvariableop_resource_0*
_output_shapes
:	Ќ*
dtype02*
(while/lstm_cell_18/MatMul/ReadVariableOpз
while/lstm_cell_18/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_18/MatMulЯ
*while/lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_18_matmul_1_readvariableop_resource_0*
_output_shapes
:	KЌ*
dtype02,
*while/lstm_cell_18/MatMul_1/ReadVariableOpР
while/lstm_cell_18/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_18/MatMul_1И
while/lstm_cell_18/addAddV2#while/lstm_cell_18/MatMul:product:0%while/lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_18/addШ
)while/lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_18_biasadd_readvariableop_resource_0*
_output_shapes	
:Ќ*
dtype02+
)while/lstm_cell_18/BiasAdd/ReadVariableOpХ
while/lstm_cell_18/BiasAddBiasAddwhile/lstm_cell_18/add:z:01while/lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_18/BiasAddv
while/lstm_cell_18/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_18/Const
"while/lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_18/split/split_dim
while/lstm_cell_18/splitSplit+while/lstm_cell_18/split/split_dim:output:0#while/lstm_cell_18/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*
	num_split2
while/lstm_cell_18/split
while/lstm_cell_18/SigmoidSigmoid!while/lstm_cell_18/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/Sigmoid
while/lstm_cell_18/Sigmoid_1Sigmoid!while/lstm_cell_18/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/Sigmoid_1 
while/lstm_cell_18/mulMul while/lstm_cell_18/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/mul
while/lstm_cell_18/ReluRelu!while/lstm_cell_18/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/ReluД
while/lstm_cell_18/mul_1Mulwhile/lstm_cell_18/Sigmoid:y:0%while/lstm_cell_18/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/mul_1Љ
while/lstm_cell_18/add_1AddV2while/lstm_cell_18/mul:z:0while/lstm_cell_18/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/add_1
while/lstm_cell_18/Sigmoid_2Sigmoid!while/lstm_cell_18/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/Sigmoid_2
while/lstm_cell_18/Relu_1Reluwhile/lstm_cell_18/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/Relu_1И
while/lstm_cell_18/mul_2Mul while/lstm_cell_18/Sigmoid_2:y:0'while/lstm_cell_18/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_18/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1т
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_18/BiasAdd/ReadVariableOp)^while/lstm_cell_18/MatMul/ReadVariableOp+^while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityѕ
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_18/BiasAdd/ReadVariableOp)^while/lstm_cell_18/MatMul/ReadVariableOp+^while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1ф
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_18/BiasAdd/ReadVariableOp)^while/lstm_cell_18/MatMul/ReadVariableOp+^while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_18/BiasAdd/ReadVariableOp)^while/lstm_cell_18/MatMul/ReadVariableOp+^while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_18/mul_2:z:0*^while/lstm_cell_18/BiasAdd/ReadVariableOp)^while/lstm_cell_18/MatMul/ReadVariableOp+^while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_18/add_1:z:0*^while/lstm_cell_18/BiasAdd/ReadVariableOp)^while/lstm_cell_18/MatMul/ReadVariableOp+^while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_18_biasadd_readvariableop_resource4while_lstm_cell_18_biasadd_readvariableop_resource_0"l
3while_lstm_cell_18_matmul_1_readvariableop_resource5while_lstm_cell_18_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_18_matmul_readvariableop_resource3while_lstm_cell_18_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџK:џџџџџџџџџK: : :::2V
)while/lstm_cell_18/BiasAdd/ReadVariableOp)while/lstm_cell_18/BiasAdd/ReadVariableOp2T
(while/lstm_cell_18/MatMul/ReadVariableOp(while/lstm_cell_18/MatMul/ReadVariableOp2X
*while/lstm_cell_18/MatMul_1/ReadVariableOp*while/lstm_cell_18/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџK:-)
'
_output_shapes
:џџџџџџџџџK:

_output_shapes
: :

_output_shapes
: 
ош
ь
E__inference_model_9_layer_call_and_return_conditional_losses_44729219

inputs7
3lstm_18_lstm_cell_18_matmul_readvariableop_resource9
5lstm_18_lstm_cell_18_matmul_1_readvariableop_resource8
4lstm_18_lstm_cell_18_biasadd_readvariableop_resource,
(gru_9_gru_cell_9_readvariableop_resource3
/gru_9_gru_cell_9_matmul_readvariableop_resource5
1gru_9_gru_cell_9_matmul_1_readvariableop_resource7
3lstm_19_lstm_cell_19_matmul_readvariableop_resource9
5lstm_19_lstm_cell_19_matmul_1_readvariableop_resource8
4lstm_19_lstm_cell_19_biasadd_readvariableop_resource+
'dense_27_matmul_readvariableop_resource,
(dense_27_biasadd_readvariableop_resource+
'dense_28_matmul_readvariableop_resource,
(dense_28_biasadd_readvariableop_resource+
'dense_29_matmul_readvariableop_resource,
(dense_29_biasadd_readvariableop_resource
identityЂdense_27/BiasAdd/ReadVariableOpЂdense_27/MatMul/ReadVariableOpЂdense_28/BiasAdd/ReadVariableOpЂdense_28/MatMul/ReadVariableOpЂdense_29/BiasAdd/ReadVariableOpЂdense_29/MatMul/ReadVariableOpЂ&gru_9/gru_cell_9/MatMul/ReadVariableOpЂ(gru_9/gru_cell_9/MatMul_1/ReadVariableOpЂgru_9/gru_cell_9/ReadVariableOpЂgru_9/whileЂ+lstm_18/lstm_cell_18/BiasAdd/ReadVariableOpЂ*lstm_18/lstm_cell_18/MatMul/ReadVariableOpЂ,lstm_18/lstm_cell_18/MatMul_1/ReadVariableOpЂlstm_18/whileЂ+lstm_19/lstm_cell_19/BiasAdd/ReadVariableOpЂ*lstm_19/lstm_cell_19/MatMul/ReadVariableOpЂ,lstm_19/lstm_cell_19/MatMul_1/ReadVariableOpЂlstm_19/whileT
lstm_18/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_18/Shape
lstm_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_18/strided_slice/stack
lstm_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_18/strided_slice/stack_1
lstm_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_18/strided_slice/stack_2
lstm_18/strided_sliceStridedSlicelstm_18/Shape:output:0$lstm_18/strided_slice/stack:output:0&lstm_18/strided_slice/stack_1:output:0&lstm_18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_18/strided_slicel
lstm_18/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
lstm_18/zeros/mul/y
lstm_18/zeros/mulMullstm_18/strided_slice:output:0lstm_18/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_18/zeros/mulo
lstm_18/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_18/zeros/Less/y
lstm_18/zeros/LessLesslstm_18/zeros/mul:z:0lstm_18/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_18/zeros/Lessr
lstm_18/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
lstm_18/zeros/packed/1Ѓ
lstm_18/zeros/packedPacklstm_18/strided_slice:output:0lstm_18/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_18/zeros/packedo
lstm_18/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_18/zeros/Const
lstm_18/zerosFilllstm_18/zeros/packed:output:0lstm_18/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_18/zerosp
lstm_18/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
lstm_18/zeros_1/mul/y
lstm_18/zeros_1/mulMullstm_18/strided_slice:output:0lstm_18/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_18/zeros_1/muls
lstm_18/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_18/zeros_1/Less/y
lstm_18/zeros_1/LessLesslstm_18/zeros_1/mul:z:0lstm_18/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_18/zeros_1/Lessv
lstm_18/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
lstm_18/zeros_1/packed/1Љ
lstm_18/zeros_1/packedPacklstm_18/strided_slice:output:0!lstm_18/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_18/zeros_1/packeds
lstm_18/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_18/zeros_1/Const
lstm_18/zeros_1Filllstm_18/zeros_1/packed:output:0lstm_18/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_18/zeros_1
lstm_18/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_18/transpose/perm
lstm_18/transpose	Transposeinputslstm_18/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
lstm_18/transposeg
lstm_18/Shape_1Shapelstm_18/transpose:y:0*
T0*
_output_shapes
:2
lstm_18/Shape_1
lstm_18/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_18/strided_slice_1/stack
lstm_18/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_18/strided_slice_1/stack_1
lstm_18/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_18/strided_slice_1/stack_2
lstm_18/strided_slice_1StridedSlicelstm_18/Shape_1:output:0&lstm_18/strided_slice_1/stack:output:0(lstm_18/strided_slice_1/stack_1:output:0(lstm_18/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_18/strided_slice_1
#lstm_18/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#lstm_18/TensorArrayV2/element_shapeв
lstm_18/TensorArrayV2TensorListReserve,lstm_18/TensorArrayV2/element_shape:output:0 lstm_18/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_18/TensorArrayV2Я
=lstm_18/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2?
=lstm_18/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_18/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_18/transpose:y:0Flstm_18/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_18/TensorArrayUnstack/TensorListFromTensor
lstm_18/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_18/strided_slice_2/stack
lstm_18/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_18/strided_slice_2/stack_1
lstm_18/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_18/strided_slice_2/stack_2Ќ
lstm_18/strided_slice_2StridedSlicelstm_18/transpose:y:0&lstm_18/strided_slice_2/stack:output:0(lstm_18/strided_slice_2/stack_1:output:0(lstm_18/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
lstm_18/strided_slice_2Э
*lstm_18/lstm_cell_18/MatMul/ReadVariableOpReadVariableOp3lstm_18_lstm_cell_18_matmul_readvariableop_resource*
_output_shapes
:	Ќ*
dtype02,
*lstm_18/lstm_cell_18/MatMul/ReadVariableOpЭ
lstm_18/lstm_cell_18/MatMulMatMul lstm_18/strided_slice_2:output:02lstm_18/lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_18/lstm_cell_18/MatMulг
,lstm_18/lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp5lstm_18_lstm_cell_18_matmul_1_readvariableop_resource*
_output_shapes
:	KЌ*
dtype02.
,lstm_18/lstm_cell_18/MatMul_1/ReadVariableOpЩ
lstm_18/lstm_cell_18/MatMul_1MatMullstm_18/zeros:output:04lstm_18/lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_18/lstm_cell_18/MatMul_1Р
lstm_18/lstm_cell_18/addAddV2%lstm_18/lstm_cell_18/MatMul:product:0'lstm_18/lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_18/lstm_cell_18/addЬ
+lstm_18/lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp4lstm_18_lstm_cell_18_biasadd_readvariableop_resource*
_output_shapes	
:Ќ*
dtype02-
+lstm_18/lstm_cell_18/BiasAdd/ReadVariableOpЭ
lstm_18/lstm_cell_18/BiasAddBiasAddlstm_18/lstm_cell_18/add:z:03lstm_18/lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_18/lstm_cell_18/BiasAddz
lstm_18/lstm_cell_18/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_18/lstm_cell_18/Const
$lstm_18/lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_18/lstm_cell_18/split/split_dim
lstm_18/lstm_cell_18/splitSplit-lstm_18/lstm_cell_18/split/split_dim:output:0%lstm_18/lstm_cell_18/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*
	num_split2
lstm_18/lstm_cell_18/split
lstm_18/lstm_cell_18/SigmoidSigmoid#lstm_18/lstm_cell_18/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_18/lstm_cell_18/SigmoidЂ
lstm_18/lstm_cell_18/Sigmoid_1Sigmoid#lstm_18/lstm_cell_18/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџK2 
lstm_18/lstm_cell_18/Sigmoid_1Ћ
lstm_18/lstm_cell_18/mulMul"lstm_18/lstm_cell_18/Sigmoid_1:y:0lstm_18/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_18/lstm_cell_18/mul
lstm_18/lstm_cell_18/ReluRelu#lstm_18/lstm_cell_18/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_18/lstm_cell_18/ReluМ
lstm_18/lstm_cell_18/mul_1Mul lstm_18/lstm_cell_18/Sigmoid:y:0'lstm_18/lstm_cell_18/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_18/lstm_cell_18/mul_1Б
lstm_18/lstm_cell_18/add_1AddV2lstm_18/lstm_cell_18/mul:z:0lstm_18/lstm_cell_18/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_18/lstm_cell_18/add_1Ђ
lstm_18/lstm_cell_18/Sigmoid_2Sigmoid#lstm_18/lstm_cell_18/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџK2 
lstm_18/lstm_cell_18/Sigmoid_2
lstm_18/lstm_cell_18/Relu_1Relulstm_18/lstm_cell_18/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_18/lstm_cell_18/Relu_1Р
lstm_18/lstm_cell_18/mul_2Mul"lstm_18/lstm_cell_18/Sigmoid_2:y:0)lstm_18/lstm_cell_18/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_18/lstm_cell_18/mul_2
%lstm_18/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   2'
%lstm_18/TensorArrayV2_1/element_shapeи
lstm_18/TensorArrayV2_1TensorListReserve.lstm_18/TensorArrayV2_1/element_shape:output:0 lstm_18/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_18/TensorArrayV2_1^
lstm_18/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_18/time
 lstm_18/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2"
 lstm_18/while/maximum_iterationsz
lstm_18/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_18/while/loop_counterъ
lstm_18/whileWhile#lstm_18/while/loop_counter:output:0)lstm_18/while/maximum_iterations:output:0lstm_18/time:output:0 lstm_18/TensorArrayV2_1:handle:0lstm_18/zeros:output:0lstm_18/zeros_1:output:0 lstm_18/strided_slice_1:output:0?lstm_18/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_18_lstm_cell_18_matmul_readvariableop_resource5lstm_18_lstm_cell_18_matmul_1_readvariableop_resource4lstm_18_lstm_cell_18_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџK:џџџџџџџџџK: : : : : *%
_read_only_resource_inputs
	
*'
bodyR
lstm_18_while_body_44728792*'
condR
lstm_18_while_cond_44728791*K
output_shapes:
8: : : : :џџџџџџџџџK:џџџџџџџџџK: : : : : *
parallel_iterations 2
lstm_18/whileХ
8lstm_18/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   2:
8lstm_18/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_18/TensorArrayV2Stack/TensorListStackTensorListStacklstm_18/while:output:3Alstm_18/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK*
element_dtype02,
*lstm_18/TensorArrayV2Stack/TensorListStack
lstm_18/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
lstm_18/strided_slice_3/stack
lstm_18/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_18/strided_slice_3/stack_1
lstm_18/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_18/strided_slice_3/stack_2Ъ
lstm_18/strided_slice_3StridedSlice3lstm_18/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_18/strided_slice_3/stack:output:0(lstm_18/strided_slice_3/stack_1:output:0(lstm_18/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџK*
shrink_axis_mask2
lstm_18/strided_slice_3
lstm_18/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_18/transpose_1/permЮ
lstm_18/transpose_1	Transpose3lstm_18/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_18/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
lstm_18/transpose_1v
lstm_18/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_18/runtimeP
gru_9/ShapeShapeinputs*
T0*
_output_shapes
:2
gru_9/Shape
gru_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_9/strided_slice/stack
gru_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_9/strided_slice/stack_1
gru_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_9/strided_slice/stack_2
gru_9/strided_sliceStridedSlicegru_9/Shape:output:0"gru_9/strided_slice/stack:output:0$gru_9/strided_slice/stack_1:output:0$gru_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_9/strided_sliceh
gru_9/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
gru_9/zeros/mul/y
gru_9/zeros/mulMulgru_9/strided_slice:output:0gru_9/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gru_9/zeros/mulk
gru_9/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
gru_9/zeros/Less/y
gru_9/zeros/LessLessgru_9/zeros/mul:z:0gru_9/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gru_9/zeros/Lessn
gru_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
gru_9/zeros/packed/1
gru_9/zeros/packedPackgru_9/strided_slice:output:0gru_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru_9/zeros/packedk
gru_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_9/zeros/Const
gru_9/zerosFillgru_9/zeros/packed:output:0gru_9/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_9/zeros
gru_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_9/transpose/perm
gru_9/transpose	Transposeinputsgru_9/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
gru_9/transposea
gru_9/Shape_1Shapegru_9/transpose:y:0*
T0*
_output_shapes
:2
gru_9/Shape_1
gru_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_9/strided_slice_1/stack
gru_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_9/strided_slice_1/stack_1
gru_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_9/strided_slice_1/stack_2
gru_9/strided_slice_1StridedSlicegru_9/Shape_1:output:0$gru_9/strided_slice_1/stack:output:0&gru_9/strided_slice_1/stack_1:output:0&gru_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_9/strided_slice_1
!gru_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2#
!gru_9/TensorArrayV2/element_shapeЪ
gru_9/TensorArrayV2TensorListReserve*gru_9/TensorArrayV2/element_shape:output:0gru_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_9/TensorArrayV2Ы
;gru_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2=
;gru_9/TensorArrayUnstack/TensorListFromTensor/element_shape
-gru_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_9/transpose:y:0Dgru_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02/
-gru_9/TensorArrayUnstack/TensorListFromTensor
gru_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_9/strided_slice_2/stack
gru_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_9/strided_slice_2/stack_1
gru_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_9/strided_slice_2/stack_2 
gru_9/strided_slice_2StridedSlicegru_9/transpose:y:0$gru_9/strided_slice_2/stack:output:0&gru_9/strided_slice_2/stack_1:output:0&gru_9/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
gru_9/strided_slice_2Ќ
gru_9/gru_cell_9/ReadVariableOpReadVariableOp(gru_9_gru_cell_9_readvariableop_resource*
_output_shapes
:	*
dtype02!
gru_9/gru_cell_9/ReadVariableOp
gru_9/gru_cell_9/unstackUnpack'gru_9/gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
gru_9/gru_cell_9/unstackС
&gru_9/gru_cell_9/MatMul/ReadVariableOpReadVariableOp/gru_9_gru_cell_9_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02(
&gru_9/gru_cell_9/MatMul/ReadVariableOpП
gru_9/gru_cell_9/MatMulMatMulgru_9/strided_slice_2:output:0.gru_9/gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_9/gru_cell_9/MatMulИ
gru_9/gru_cell_9/BiasAddBiasAdd!gru_9/gru_cell_9/MatMul:product:0!gru_9/gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_9/gru_cell_9/BiasAddr
gru_9/gru_cell_9/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_9/gru_cell_9/Const
 gru_9/gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2"
 gru_9/gru_cell_9/split/split_dim№
gru_9/gru_cell_9/splitSplit)gru_9/gru_cell_9/split/split_dim:output:0!gru_9/gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
gru_9/gru_cell_9/splitЧ
(gru_9/gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp1gru_9_gru_cell_9_matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype02*
(gru_9/gru_cell_9/MatMul_1/ReadVariableOpЛ
gru_9/gru_cell_9/MatMul_1MatMulgru_9/zeros:output:00gru_9/gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_9/gru_cell_9/MatMul_1О
gru_9/gru_cell_9/BiasAdd_1BiasAdd#gru_9/gru_cell_9/MatMul_1:product:0!gru_9/gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_9/gru_cell_9/BiasAdd_1
gru_9/gru_cell_9/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   џџџџ2
gru_9/gru_cell_9/Const_1
"gru_9/gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2$
"gru_9/gru_cell_9/split_1/split_dimЈ
gru_9/gru_cell_9/split_1SplitV#gru_9/gru_cell_9/BiasAdd_1:output:0!gru_9/gru_cell_9/Const_1:output:0+gru_9/gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
gru_9/gru_cell_9/split_1Ћ
gru_9/gru_cell_9/addAddV2gru_9/gru_cell_9/split:output:0!gru_9/gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_9/gru_cell_9/add
gru_9/gru_cell_9/SigmoidSigmoidgru_9/gru_cell_9/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_9/gru_cell_9/SigmoidЏ
gru_9/gru_cell_9/add_1AddV2gru_9/gru_cell_9/split:output:1!gru_9/gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_9/gru_cell_9/add_1
gru_9/gru_cell_9/Sigmoid_1Sigmoidgru_9/gru_cell_9/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_9/gru_cell_9/Sigmoid_1Ј
gru_9/gru_cell_9/mulMulgru_9/gru_cell_9/Sigmoid_1:y:0!gru_9/gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_9/gru_cell_9/mulІ
gru_9/gru_cell_9/add_2AddV2gru_9/gru_cell_9/split:output:2gru_9/gru_cell_9/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_9/gru_cell_9/add_2
gru_9/gru_cell_9/ReluRelugru_9/gru_cell_9/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_9/gru_cell_9/Relu
gru_9/gru_cell_9/mul_1Mulgru_9/gru_cell_9/Sigmoid:y:0gru_9/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_9/gru_cell_9/mul_1u
gru_9/gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_9/gru_cell_9/sub/xЄ
gru_9/gru_cell_9/subSubgru_9/gru_cell_9/sub/x:output:0gru_9/gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_9/gru_cell_9/subЈ
gru_9/gru_cell_9/mul_2Mulgru_9/gru_cell_9/sub:z:0#gru_9/gru_cell_9/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_9/gru_cell_9/mul_2Ѓ
gru_9/gru_cell_9/add_3AddV2gru_9/gru_cell_9/mul_1:z:0gru_9/gru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_9/gru_cell_9/add_3
#gru_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2%
#gru_9/TensorArrayV2_1/element_shapeа
gru_9/TensorArrayV2_1TensorListReserve,gru_9/TensorArrayV2_1/element_shape:output:0gru_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_9/TensorArrayV2_1Z

gru_9/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

gru_9/time
gru_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2 
gru_9/while/maximum_iterationsv
gru_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_9/while/loop_counterџ
gru_9/whileWhile!gru_9/while/loop_counter:output:0'gru_9/while/maximum_iterations:output:0gru_9/time:output:0gru_9/TensorArrayV2_1:handle:0gru_9/zeros:output:0gru_9/strided_slice_1:output:0=gru_9/TensorArrayUnstack/TensorListFromTensor:output_handle:0(gru_9_gru_cell_9_readvariableop_resource/gru_9_gru_cell_9_matmul_readvariableop_resource1gru_9_gru_cell_9_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	*%
bodyR
gru_9_while_body_44728942*%
condR
gru_9_while_cond_44728941*8
output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *
parallel_iterations 2
gru_9/whileС
6gru_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   28
6gru_9/TensorArrayV2Stack/TensorListStack/element_shape
(gru_9/TensorArrayV2Stack/TensorListStackTensorListStackgru_9/while:output:3?gru_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02*
(gru_9/TensorArrayV2Stack/TensorListStack
gru_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
gru_9/strided_slice_3/stack
gru_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
gru_9/strided_slice_3/stack_1
gru_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_9/strided_slice_3/stack_2О
gru_9/strided_slice_3StridedSlice1gru_9/TensorArrayV2Stack/TensorListStack:tensor:0$gru_9/strided_slice_3/stack:output:0&gru_9/strided_slice_3/stack_1:output:0&gru_9/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
gru_9/strided_slice_3
gru_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_9/transpose_1/permЦ
gru_9/transpose_1	Transpose1gru_9/TensorArrayV2Stack/TensorListStack:tensor:0gru_9/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
gru_9/transpose_1r
gru_9/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_9/runtimey
dropout_18/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЂМ?2
dropout_18/dropout/ConstВ
dropout_18/dropout/MulMullstm_18/transpose_1:y:0!dropout_18/dropout/Const:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
dropout_18/dropout/Mul{
dropout_18/dropout/ShapeShapelstm_18/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_18/dropout/Shapeт
/dropout_18/dropout/random_uniform/RandomUniformRandomUniform!dropout_18/dropout/Shape:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK*
dtype021
/dropout_18/dropout/random_uniform/RandomUniform
!dropout_18/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL=2#
!dropout_18/dropout/GreaterEqual/yї
dropout_18/dropout/GreaterEqualGreaterEqual8dropout_18/dropout/random_uniform/RandomUniform:output:0*dropout_18/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2!
dropout_18/dropout/GreaterEqual­
dropout_18/dropout/CastCast#dropout_18/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
dropout_18/dropout/CastГ
dropout_18/dropout/Mul_1Muldropout_18/dropout/Mul:z:0dropout_18/dropout/Cast:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
dropout_18/dropout/Mul_1y
dropout_19/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout_19/dropout/ConstЌ
dropout_19/dropout/MulMulgru_9/strided_slice_3:output:0!dropout_19/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
dropout_19/dropout/Mul
dropout_19/dropout/ShapeShapegru_9/strided_slice_3:output:0*
T0*
_output_shapes
:2
dropout_19/dropout/Shapeе
/dropout_19/dropout/random_uniform/RandomUniformRandomUniform!dropout_19/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
dtype021
/dropout_19/dropout/random_uniform/RandomUniform
!dropout_19/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dropout_19/dropout/GreaterEqual/yъ
dropout_19/dropout/GreaterEqualGreaterEqual8dropout_19/dropout/random_uniform/RandomUniform:output:0*dropout_19/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22!
dropout_19/dropout/GreaterEqual 
dropout_19/dropout/CastCast#dropout_19/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ22
dropout_19/dropout/CastІ
dropout_19/dropout/Mul_1Muldropout_19/dropout/Mul:z:0dropout_19/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
dropout_19/dropout/Mul_1j
lstm_19/ShapeShapedropout_18/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
lstm_19/Shape
lstm_19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_19/strided_slice/stack
lstm_19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_19/strided_slice/stack_1
lstm_19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_19/strided_slice/stack_2
lstm_19/strided_sliceStridedSlicelstm_19/Shape:output:0$lstm_19/strided_slice/stack:output:0&lstm_19/strided_slice/stack_1:output:0&lstm_19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_19/strided_slicel
lstm_19/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
lstm_19/zeros/mul/y
lstm_19/zeros/mulMullstm_19/strided_slice:output:0lstm_19/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_19/zeros/mulo
lstm_19/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_19/zeros/Less/y
lstm_19/zeros/LessLesslstm_19/zeros/mul:z:0lstm_19/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_19/zeros/Lessr
lstm_19/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
lstm_19/zeros/packed/1Ѓ
lstm_19/zeros/packedPacklstm_19/strided_slice:output:0lstm_19/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_19/zeros/packedo
lstm_19/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_19/zeros/Const
lstm_19/zerosFilllstm_19/zeros/packed:output:0lstm_19/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_19/zerosp
lstm_19/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
lstm_19/zeros_1/mul/y
lstm_19/zeros_1/mulMullstm_19/strided_slice:output:0lstm_19/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_19/zeros_1/muls
lstm_19/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_19/zeros_1/Less/y
lstm_19/zeros_1/LessLesslstm_19/zeros_1/mul:z:0lstm_19/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_19/zeros_1/Lessv
lstm_19/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
lstm_19/zeros_1/packed/1Љ
lstm_19/zeros_1/packedPacklstm_19/strided_slice:output:0!lstm_19/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_19/zeros_1/packeds
lstm_19/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_19/zeros_1/Const
lstm_19/zeros_1Filllstm_19/zeros_1/packed:output:0lstm_19/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_19/zeros_1
lstm_19/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_19/transpose/permБ
lstm_19/transpose	Transposedropout_18/dropout/Mul_1:z:0lstm_19/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
lstm_19/transposeg
lstm_19/Shape_1Shapelstm_19/transpose:y:0*
T0*
_output_shapes
:2
lstm_19/Shape_1
lstm_19/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_19/strided_slice_1/stack
lstm_19/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_19/strided_slice_1/stack_1
lstm_19/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_19/strided_slice_1/stack_2
lstm_19/strided_slice_1StridedSlicelstm_19/Shape_1:output:0&lstm_19/strided_slice_1/stack:output:0(lstm_19/strided_slice_1/stack_1:output:0(lstm_19/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_19/strided_slice_1
#lstm_19/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#lstm_19/TensorArrayV2/element_shapeв
lstm_19/TensorArrayV2TensorListReserve,lstm_19/TensorArrayV2/element_shape:output:0 lstm_19/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_19/TensorArrayV2Я
=lstm_19/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   2?
=lstm_19/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_19/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_19/transpose:y:0Flstm_19/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_19/TensorArrayUnstack/TensorListFromTensor
lstm_19/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_19/strided_slice_2/stack
lstm_19/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_19/strided_slice_2/stack_1
lstm_19/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_19/strided_slice_2/stack_2Ќ
lstm_19/strided_slice_2StridedSlicelstm_19/transpose:y:0&lstm_19/strided_slice_2/stack:output:0(lstm_19/strided_slice_2/stack_1:output:0(lstm_19/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџK*
shrink_axis_mask2
lstm_19/strided_slice_2Э
*lstm_19/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp3lstm_19_lstm_cell_19_matmul_readvariableop_resource*
_output_shapes
:	KШ*
dtype02,
*lstm_19/lstm_cell_19/MatMul/ReadVariableOpЭ
lstm_19/lstm_cell_19/MatMulMatMul lstm_19/strided_slice_2:output:02lstm_19/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_19/lstm_cell_19/MatMulг
,lstm_19/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp5lstm_19_lstm_cell_19_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02.
,lstm_19/lstm_cell_19/MatMul_1/ReadVariableOpЩ
lstm_19/lstm_cell_19/MatMul_1MatMullstm_19/zeros:output:04lstm_19/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_19/lstm_cell_19/MatMul_1Р
lstm_19/lstm_cell_19/addAddV2%lstm_19/lstm_cell_19/MatMul:product:0'lstm_19/lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_19/lstm_cell_19/addЬ
+lstm_19/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp4lstm_19_lstm_cell_19_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02-
+lstm_19/lstm_cell_19/BiasAdd/ReadVariableOpЭ
lstm_19/lstm_cell_19/BiasAddBiasAddlstm_19/lstm_cell_19/add:z:03lstm_19/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_19/lstm_cell_19/BiasAddz
lstm_19/lstm_cell_19/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_19/lstm_cell_19/Const
$lstm_19/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_19/lstm_cell_19/split/split_dim
lstm_19/lstm_cell_19/splitSplit-lstm_19/lstm_cell_19/split/split_dim:output:0%lstm_19/lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_19/lstm_cell_19/split
lstm_19/lstm_cell_19/SigmoidSigmoid#lstm_19/lstm_cell_19/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_19/lstm_cell_19/SigmoidЂ
lstm_19/lstm_cell_19/Sigmoid_1Sigmoid#lstm_19/lstm_cell_19/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22 
lstm_19/lstm_cell_19/Sigmoid_1Ћ
lstm_19/lstm_cell_19/mulMul"lstm_19/lstm_cell_19/Sigmoid_1:y:0lstm_19/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_19/lstm_cell_19/mul
lstm_19/lstm_cell_19/ReluRelu#lstm_19/lstm_cell_19/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_19/lstm_cell_19/ReluМ
lstm_19/lstm_cell_19/mul_1Mul lstm_19/lstm_cell_19/Sigmoid:y:0'lstm_19/lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_19/lstm_cell_19/mul_1Б
lstm_19/lstm_cell_19/add_1AddV2lstm_19/lstm_cell_19/mul:z:0lstm_19/lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_19/lstm_cell_19/add_1Ђ
lstm_19/lstm_cell_19/Sigmoid_2Sigmoid#lstm_19/lstm_cell_19/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22 
lstm_19/lstm_cell_19/Sigmoid_2
lstm_19/lstm_cell_19/Relu_1Relulstm_19/lstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_19/lstm_cell_19/Relu_1Р
lstm_19/lstm_cell_19/mul_2Mul"lstm_19/lstm_cell_19/Sigmoid_2:y:0)lstm_19/lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_19/lstm_cell_19/mul_2
%lstm_19/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2'
%lstm_19/TensorArrayV2_1/element_shapeи
lstm_19/TensorArrayV2_1TensorListReserve.lstm_19/TensorArrayV2_1/element_shape:output:0 lstm_19/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_19/TensorArrayV2_1^
lstm_19/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_19/time
 lstm_19/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2"
 lstm_19/while/maximum_iterationsz
lstm_19/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_19/while/loop_counterъ
lstm_19/whileWhile#lstm_19/while/loop_counter:output:0)lstm_19/while/maximum_iterations:output:0lstm_19/time:output:0 lstm_19/TensorArrayV2_1:handle:0lstm_19/zeros:output:0lstm_19/zeros_1:output:0 lstm_19/strided_slice_1:output:0?lstm_19/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_19_lstm_cell_19_matmul_readvariableop_resource5lstm_19_lstm_cell_19_matmul_1_readvariableop_resource4lstm_19_lstm_cell_19_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*'
bodyR
lstm_19_while_body_44729112*'
condR
lstm_19_while_cond_44729111*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
lstm_19/whileХ
8lstm_19/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2:
8lstm_19/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_19/TensorArrayV2Stack/TensorListStackTensorListStacklstm_19/while:output:3Alstm_19/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02,
*lstm_19/TensorArrayV2Stack/TensorListStack
lstm_19/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
lstm_19/strided_slice_3/stack
lstm_19/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_19/strided_slice_3/stack_1
lstm_19/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_19/strided_slice_3/stack_2Ъ
lstm_19/strided_slice_3StridedSlice3lstm_19/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_19/strided_slice_3/stack:output:0(lstm_19/strided_slice_3/stack_1:output:0(lstm_19/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
lstm_19/strided_slice_3
lstm_19/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_19/transpose_1/permЮ
lstm_19/transpose_1	Transpose3lstm_19/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_19/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
lstm_19/transpose_1v
lstm_19/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_19/runtimeЈ
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*
_output_shapes

:2 *
dtype02 
dense_27/MatMul/ReadVariableOpЈ
dense_27/MatMulMatMul lstm_19/strided_slice_3:output:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_27/MatMulЇ
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_27/BiasAdd/ReadVariableOpЅ
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_27/BiasAdds
dense_27/ReluReludense_27/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_27/ReluЈ
dense_28/MatMul/ReadVariableOpReadVariableOp'dense_28_matmul_readvariableop_resource*
_output_shapes

:2@*
dtype02 
dense_28/MatMul/ReadVariableOpЄ
dense_28/MatMulMatMuldropout_19/dropout/Mul_1:z:0&dense_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_28/MatMulЇ
dense_28/BiasAdd/ReadVariableOpReadVariableOp(dense_28_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_28/BiasAdd/ReadVariableOpЅ
dense_28/BiasAddBiasAdddense_28/MatMul:product:0'dense_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_28/BiasAdds
dense_28/ReluReludense_28/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_28/Relux
concatenate_9/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_9/concat/axisб
concatenate_9/concatConcatV2dense_27/Relu:activations:0dense_28/Relu:activations:0"concatenate_9/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ`2
concatenate_9/concatЈ
dense_29/MatMul/ReadVariableOpReadVariableOp'dense_29_matmul_readvariableop_resource*
_output_shapes

:`*
dtype02 
dense_29/MatMul/ReadVariableOpЅ
dense_29/MatMulMatMulconcatenate_9/concat:output:0&dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_29/MatMulЇ
dense_29/BiasAdd/ReadVariableOpReadVariableOp(dense_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_29/BiasAdd/ReadVariableOpЅ
dense_29/BiasAddBiasAdddense_29/MatMul:product:0'dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_29/BiasAddю
IdentityIdentitydense_29/BiasAdd:output:0 ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp ^dense_28/BiasAdd/ReadVariableOp^dense_28/MatMul/ReadVariableOp ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOp'^gru_9/gru_cell_9/MatMul/ReadVariableOp)^gru_9/gru_cell_9/MatMul_1/ReadVariableOp ^gru_9/gru_cell_9/ReadVariableOp^gru_9/while,^lstm_18/lstm_cell_18/BiasAdd/ReadVariableOp+^lstm_18/lstm_cell_18/MatMul/ReadVariableOp-^lstm_18/lstm_cell_18/MatMul_1/ReadVariableOp^lstm_18/while,^lstm_19/lstm_cell_19/BiasAdd/ReadVariableOp+^lstm_19/lstm_cell_19/MatMul/ReadVariableOp-^lstm_19/lstm_cell_19/MatMul_1/ReadVariableOp^lstm_19/while*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:џџџџџџџџџџџџџџџџџџ:::::::::::::::2B
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp2B
dense_28/BiasAdd/ReadVariableOpdense_28/BiasAdd/ReadVariableOp2@
dense_28/MatMul/ReadVariableOpdense_28/MatMul/ReadVariableOp2B
dense_29/BiasAdd/ReadVariableOpdense_29/BiasAdd/ReadVariableOp2@
dense_29/MatMul/ReadVariableOpdense_29/MatMul/ReadVariableOp2P
&gru_9/gru_cell_9/MatMul/ReadVariableOp&gru_9/gru_cell_9/MatMul/ReadVariableOp2T
(gru_9/gru_cell_9/MatMul_1/ReadVariableOp(gru_9/gru_cell_9/MatMul_1/ReadVariableOp2B
gru_9/gru_cell_9/ReadVariableOpgru_9/gru_cell_9/ReadVariableOp2
gru_9/whilegru_9/while2Z
+lstm_18/lstm_cell_18/BiasAdd/ReadVariableOp+lstm_18/lstm_cell_18/BiasAdd/ReadVariableOp2X
*lstm_18/lstm_cell_18/MatMul/ReadVariableOp*lstm_18/lstm_cell_18/MatMul/ReadVariableOp2\
,lstm_18/lstm_cell_18/MatMul_1/ReadVariableOp,lstm_18/lstm_cell_18/MatMul_1/ReadVariableOp2
lstm_18/whilelstm_18/while2Z
+lstm_19/lstm_cell_19/BiasAdd/ReadVariableOp+lstm_19/lstm_cell_19/BiasAdd/ReadVariableOp2X
*lstm_19/lstm_cell_19/MatMul/ReadVariableOp*lstm_19/lstm_cell_19/MatMul/ReadVariableOp2\
,lstm_19/lstm_cell_19/MatMul_1/ReadVariableOp,lstm_19/lstm_cell_19/MatMul_1/ReadVariableOp2
lstm_19/whilelstm_19/while:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
џF
Ї
while_body_44727867
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
*while_gru_cell_9_readvariableop_resource_05
1while_gru_cell_9_matmul_readvariableop_resource_07
3while_gru_cell_9_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
(while_gru_cell_9_readvariableop_resource3
/while_gru_cell_9_matmul_readvariableop_resource5
1while_gru_cell_9_matmul_1_readvariableop_resourceЂ&while/gru_cell_9/MatMul/ReadVariableOpЂ(while/gru_cell_9/MatMul_1/ReadVariableOpЂwhile/gru_cell_9/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЎ
while/gru_cell_9/ReadVariableOpReadVariableOp*while_gru_cell_9_readvariableop_resource_0*
_output_shapes
:	*
dtype02!
while/gru_cell_9/ReadVariableOp
while/gru_cell_9/unstackUnpack'while/gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
while/gru_cell_9/unstackУ
&while/gru_cell_9/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/gru_cell_9/MatMul/ReadVariableOpб
while/gru_cell_9/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
while/gru_cell_9/MatMulИ
while/gru_cell_9/BiasAddBiasAdd!while/gru_cell_9/MatMul:product:0!while/gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
while/gru_cell_9/BiasAddr
while/gru_cell_9/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_9/Const
 while/gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2"
 while/gru_cell_9/split/split_dim№
while/gru_cell_9/splitSplit)while/gru_cell_9/split/split_dim:output:0!while/gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/gru_cell_9/splitЩ
(while/gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes
:	2*
dtype02*
(while/gru_cell_9/MatMul_1/ReadVariableOpК
while/gru_cell_9/MatMul_1MatMulwhile_placeholder_20while/gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
while/gru_cell_9/MatMul_1О
while/gru_cell_9/BiasAdd_1BiasAdd#while/gru_cell_9/MatMul_1:product:0!while/gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2
while/gru_cell_9/BiasAdd_1
while/gru_cell_9/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   џџџџ2
while/gru_cell_9/Const_1
"while/gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2$
"while/gru_cell_9/split_1/split_dimЈ
while/gru_cell_9/split_1SplitV#while/gru_cell_9/BiasAdd_1:output:0!while/gru_cell_9/Const_1:output:0+while/gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/gru_cell_9/split_1Ћ
while/gru_cell_9/addAddV2while/gru_cell_9/split:output:0!while/gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/add
while/gru_cell_9/SigmoidSigmoidwhile/gru_cell_9/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/SigmoidЏ
while/gru_cell_9/add_1AddV2while/gru_cell_9/split:output:1!while/gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/add_1
while/gru_cell_9/Sigmoid_1Sigmoidwhile/gru_cell_9/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/Sigmoid_1Ј
while/gru_cell_9/mulMulwhile/gru_cell_9/Sigmoid_1:y:0!while/gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/mulІ
while/gru_cell_9/add_2AddV2while/gru_cell_9/split:output:2while/gru_cell_9/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/add_2
while/gru_cell_9/ReluReluwhile/gru_cell_9/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/Relu
while/gru_cell_9/mul_1Mulwhile/gru_cell_9/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/mul_1u
while/gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_9/sub/xЄ
while/gru_cell_9/subSubwhile/gru_cell_9/sub/x:output:0while/gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/subЈ
while/gru_cell_9/mul_2Mulwhile/gru_cell_9/sub:z:0#while/gru_cell_9/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/mul_2Ѓ
while/gru_cell_9/add_3AddV2while/gru_cell_9/mul_1:z:0while/gru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/add_3о
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_9/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1д
while/IdentityIdentitywhile/add_1:z:0'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityч
while/Identity_1Identitywhile_while_maximum_iterations'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1ж
while/Identity_2Identitywhile/add:z:0'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3є
while/Identity_4Identitywhile/gru_cell_9/add_3:z:0'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4"h
1while_gru_cell_9_matmul_1_readvariableop_resource3while_gru_cell_9_matmul_1_readvariableop_resource_0"d
/while_gru_cell_9_matmul_readvariableop_resource1while_gru_cell_9_matmul_readvariableop_resource_0"V
(while_gru_cell_9_readvariableop_resource*while_gru_cell_9_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ2: : :::2P
&while/gru_cell_9/MatMul/ReadVariableOp&while/gru_cell_9/MatMul/ReadVariableOp2T
(while/gru_cell_9/MatMul_1/ReadVariableOp(while/gru_cell_9/MatMul_1/ReadVariableOp2B
while/gru_cell_9/ReadVariableOpwhile/gru_cell_9/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
Ж

*__inference_lstm_18_layer_call_fn_44730087

inputs
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_lstm_18_layer_call_and_return_conditional_losses_447274572
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Е
п
J__inference_lstm_cell_18_layer_call_and_return_conditional_losses_44731954

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ќ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	KЌ*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ќ*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimП
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџK2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџK2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџK2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџK2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
mul_2Ј
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2

IdentityЌ

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2

Identity_1Ќ

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:џџџџџџџџџ:џџџџџџџџџK:џџџџџџџџџK:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџK
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџK
"
_user_specified_name
states/1
Ы
f
H__inference_dropout_19_layer_call_and_return_conditional_losses_44731806

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ22

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:џџџџџџџџџ2:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
џF
Ї
while_body_44731021
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
*while_gru_cell_9_readvariableop_resource_05
1while_gru_cell_9_matmul_readvariableop_resource_07
3while_gru_cell_9_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
(while_gru_cell_9_readvariableop_resource3
/while_gru_cell_9_matmul_readvariableop_resource5
1while_gru_cell_9_matmul_1_readvariableop_resourceЂ&while/gru_cell_9/MatMul/ReadVariableOpЂ(while/gru_cell_9/MatMul_1/ReadVariableOpЂwhile/gru_cell_9/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЎ
while/gru_cell_9/ReadVariableOpReadVariableOp*while_gru_cell_9_readvariableop_resource_0*
_output_shapes
:	*
dtype02!
while/gru_cell_9/ReadVariableOp
while/gru_cell_9/unstackUnpack'while/gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
while/gru_cell_9/unstackУ
&while/gru_cell_9/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/gru_cell_9/MatMul/ReadVariableOpб
while/gru_cell_9/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
while/gru_cell_9/MatMulИ
while/gru_cell_9/BiasAddBiasAdd!while/gru_cell_9/MatMul:product:0!while/gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
while/gru_cell_9/BiasAddr
while/gru_cell_9/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_9/Const
 while/gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2"
 while/gru_cell_9/split/split_dim№
while/gru_cell_9/splitSplit)while/gru_cell_9/split/split_dim:output:0!while/gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/gru_cell_9/splitЩ
(while/gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes
:	2*
dtype02*
(while/gru_cell_9/MatMul_1/ReadVariableOpК
while/gru_cell_9/MatMul_1MatMulwhile_placeholder_20while/gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
while/gru_cell_9/MatMul_1О
while/gru_cell_9/BiasAdd_1BiasAdd#while/gru_cell_9/MatMul_1:product:0!while/gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2
while/gru_cell_9/BiasAdd_1
while/gru_cell_9/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   џџџџ2
while/gru_cell_9/Const_1
"while/gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2$
"while/gru_cell_9/split_1/split_dimЈ
while/gru_cell_9/split_1SplitV#while/gru_cell_9/BiasAdd_1:output:0!while/gru_cell_9/Const_1:output:0+while/gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/gru_cell_9/split_1Ћ
while/gru_cell_9/addAddV2while/gru_cell_9/split:output:0!while/gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/add
while/gru_cell_9/SigmoidSigmoidwhile/gru_cell_9/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/SigmoidЏ
while/gru_cell_9/add_1AddV2while/gru_cell_9/split:output:1!while/gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/add_1
while/gru_cell_9/Sigmoid_1Sigmoidwhile/gru_cell_9/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/Sigmoid_1Ј
while/gru_cell_9/mulMulwhile/gru_cell_9/Sigmoid_1:y:0!while/gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/mulІ
while/gru_cell_9/add_2AddV2while/gru_cell_9/split:output:2while/gru_cell_9/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/add_2
while/gru_cell_9/ReluReluwhile/gru_cell_9/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/Relu
while/gru_cell_9/mul_1Mulwhile/gru_cell_9/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/mul_1u
while/gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_9/sub/xЄ
while/gru_cell_9/subSubwhile/gru_cell_9/sub/x:output:0while/gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/subЈ
while/gru_cell_9/mul_2Mulwhile/gru_cell_9/sub:z:0#while/gru_cell_9/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/mul_2Ѓ
while/gru_cell_9/add_3AddV2while/gru_cell_9/mul_1:z:0while/gru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/add_3о
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_9/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1д
while/IdentityIdentitywhile/add_1:z:0'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityч
while/Identity_1Identitywhile_while_maximum_iterations'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1ж
while/Identity_2Identitywhile/add:z:0'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3є
while/Identity_4Identitywhile/gru_cell_9/add_3:z:0'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4"h
1while_gru_cell_9_matmul_1_readvariableop_resource3while_gru_cell_9_matmul_1_readvariableop_resource_0"d
/while_gru_cell_9_matmul_readvariableop_resource1while_gru_cell_9_matmul_readvariableop_resource_0"V
(while_gru_cell_9_readvariableop_resource*while_gru_cell_9_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ2: : :::2P
&while/gru_cell_9/MatMul/ReadVariableOp&while/gru_cell_9/MatMul/ReadVariableOp2T
(while/gru_cell_9/MatMul_1/ReadVariableOp(while/gru_cell_9/MatMul_1/ReadVariableOp2B
while/gru_cell_9/ReadVariableOpwhile/gru_cell_9/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
щZ
ж
C__inference_gru_9_layer_call_and_return_conditional_losses_44730771

inputs&
"gru_cell_9_readvariableop_resource-
)gru_cell_9_matmul_readvariableop_resource/
+gru_cell_9_matmul_1_readvariableop_resource
identityЂ gru_cell_9/MatMul/ReadVariableOpЂ"gru_cell_9/MatMul_1/ReadVariableOpЂgru_cell_9/ReadVariableOpЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2
gru_cell_9/ReadVariableOpReadVariableOp"gru_cell_9_readvariableop_resource*
_output_shapes
:	*
dtype02
gru_cell_9/ReadVariableOp
gru_cell_9/unstackUnpack!gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
gru_cell_9/unstackЏ
 gru_cell_9/MatMul/ReadVariableOpReadVariableOp)gru_cell_9_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 gru_cell_9/MatMul/ReadVariableOpЇ
gru_cell_9/MatMulMatMulstrided_slice_2:output:0(gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_cell_9/MatMul 
gru_cell_9/BiasAddBiasAddgru_cell_9/MatMul:product:0gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_cell_9/BiasAddf
gru_cell_9/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_9/Const
gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_9/split/split_dimи
gru_cell_9/splitSplit#gru_cell_9/split/split_dim:output:0gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
gru_cell_9/splitЕ
"gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_9_matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype02$
"gru_cell_9/MatMul_1/ReadVariableOpЃ
gru_cell_9/MatMul_1MatMulzeros:output:0*gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_cell_9/MatMul_1І
gru_cell_9/BiasAdd_1BiasAddgru_cell_9/MatMul_1:product:0gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_cell_9/BiasAdd_1}
gru_cell_9/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   џџџџ2
gru_cell_9/Const_1
gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_9/split_1/split_dim
gru_cell_9/split_1SplitVgru_cell_9/BiasAdd_1:output:0gru_cell_9/Const_1:output:0%gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
gru_cell_9/split_1
gru_cell_9/addAddV2gru_cell_9/split:output:0gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/addy
gru_cell_9/SigmoidSigmoidgru_cell_9/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/Sigmoid
gru_cell_9/add_1AddV2gru_cell_9/split:output:1gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/add_1
gru_cell_9/Sigmoid_1Sigmoidgru_cell_9/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/Sigmoid_1
gru_cell_9/mulMulgru_cell_9/Sigmoid_1:y:0gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/mul
gru_cell_9/add_2AddV2gru_cell_9/split:output:2gru_cell_9/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/add_2r
gru_cell_9/ReluRelugru_cell_9/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/Relu
gru_cell_9/mul_1Mulgru_cell_9/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/mul_1i
gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_9/sub/x
gru_cell_9/subSubgru_cell_9/sub/x:output:0gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/sub
gru_cell_9/mul_2Mulgru_cell_9/sub:z:0gru_cell_9/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/mul_2
gru_cell_9/add_3AddV2gru_cell_9/mul_1:z:0gru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЋ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_9_readvariableop_resource)gru_cell_9_matmul_readvariableop_resource+gru_cell_9_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_44730681*
condR
while_cond_44730680*8
output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeи
IdentityIdentitystrided_slice_3:output:0!^gru_cell_9/MatMul/ReadVariableOp#^gru_cell_9/MatMul_1/ReadVariableOp^gru_cell_9/ReadVariableOp^while*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2D
 gru_cell_9/MatMul/ReadVariableOp gru_cell_9/MatMul/ReadVariableOp2H
"gru_cell_9/MatMul_1/ReadVariableOp"gru_cell_9/MatMul_1/ReadVariableOp26
gru_cell_9/ReadVariableOpgru_cell_9/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ёZ
и
C__inference_gru_9_layer_call_and_return_conditional_losses_44731111
inputs_0&
"gru_cell_9_readvariableop_resource-
)gru_cell_9_matmul_readvariableop_resource/
+gru_cell_9_matmul_1_readvariableop_resource
identityЂ gru_cell_9/MatMul/ReadVariableOpЂ"gru_cell_9/MatMul_1/ReadVariableOpЂgru_cell_9/ReadVariableOpЂwhileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2
gru_cell_9/ReadVariableOpReadVariableOp"gru_cell_9_readvariableop_resource*
_output_shapes
:	*
dtype02
gru_cell_9/ReadVariableOp
gru_cell_9/unstackUnpack!gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
gru_cell_9/unstackЏ
 gru_cell_9/MatMul/ReadVariableOpReadVariableOp)gru_cell_9_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 gru_cell_9/MatMul/ReadVariableOpЇ
gru_cell_9/MatMulMatMulstrided_slice_2:output:0(gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_cell_9/MatMul 
gru_cell_9/BiasAddBiasAddgru_cell_9/MatMul:product:0gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_cell_9/BiasAddf
gru_cell_9/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_9/Const
gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_9/split/split_dimи
gru_cell_9/splitSplit#gru_cell_9/split/split_dim:output:0gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
gru_cell_9/splitЕ
"gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_9_matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype02$
"gru_cell_9/MatMul_1/ReadVariableOpЃ
gru_cell_9/MatMul_1MatMulzeros:output:0*gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_cell_9/MatMul_1І
gru_cell_9/BiasAdd_1BiasAddgru_cell_9/MatMul_1:product:0gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_cell_9/BiasAdd_1}
gru_cell_9/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   џџџџ2
gru_cell_9/Const_1
gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_9/split_1/split_dim
gru_cell_9/split_1SplitVgru_cell_9/BiasAdd_1:output:0gru_cell_9/Const_1:output:0%gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
gru_cell_9/split_1
gru_cell_9/addAddV2gru_cell_9/split:output:0gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/addy
gru_cell_9/SigmoidSigmoidgru_cell_9/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/Sigmoid
gru_cell_9/add_1AddV2gru_cell_9/split:output:1gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/add_1
gru_cell_9/Sigmoid_1Sigmoidgru_cell_9/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/Sigmoid_1
gru_cell_9/mulMulgru_cell_9/Sigmoid_1:y:0gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/mul
gru_cell_9/add_2AddV2gru_cell_9/split:output:2gru_cell_9/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/add_2r
gru_cell_9/ReluRelugru_cell_9/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/Relu
gru_cell_9/mul_1Mulgru_cell_9/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/mul_1i
gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_9/sub/x
gru_cell_9/subSubgru_cell_9/sub/x:output:0gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/sub
gru_cell_9/mul_2Mulgru_cell_9/sub:z:0gru_cell_9/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/mul_2
gru_cell_9/add_3AddV2gru_cell_9/mul_1:z:0gru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЋ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_9_readvariableop_resource)gru_cell_9_matmul_readvariableop_resource+gru_cell_9_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_44731021*
condR
while_cond_44731020*8
output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeи
IdentityIdentitystrided_slice_3:output:0!^gru_cell_9/MatMul/ReadVariableOp#^gru_cell_9/MatMul_1/ReadVariableOp^gru_cell_9/ReadVariableOp^while*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2D
 gru_cell_9/MatMul/ReadVariableOp gru_cell_9/MatMul/ReadVariableOp2H
"gru_cell_9/MatMul_1/ReadVariableOp"gru_cell_9/MatMul_1/ReadVariableOp26
gru_cell_9/ReadVariableOpgru_cell_9/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
Е
Э
while_cond_44728266
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_44728266___redundant_placeholder06
2while_while_cond_44728266___redundant_placeholder16
2while_while_cond_44728266___redundant_placeholder26
2while_while_cond_44728266___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџ2:џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
	
І
gru_9_while_cond_44728941(
$gru_9_while_gru_9_while_loop_counter.
*gru_9_while_gru_9_while_maximum_iterations
gru_9_while_placeholder
gru_9_while_placeholder_1
gru_9_while_placeholder_2*
&gru_9_while_less_gru_9_strided_slice_1B
>gru_9_while_gru_9_while_cond_44728941___redundant_placeholder0B
>gru_9_while_gru_9_while_cond_44728941___redundant_placeholder1B
>gru_9_while_gru_9_while_cond_44728941___redundant_placeholder2B
>gru_9_while_gru_9_while_cond_44728941___redundant_placeholder3
gru_9_while_identity

gru_9/while/LessLessgru_9_while_placeholder&gru_9_while_less_gru_9_strided_slice_1*
T0*
_output_shapes
: 2
gru_9/while/Lesso
gru_9/while/IdentityIdentitygru_9/while/Less:z:0*
T0
*
_output_shapes
: 2
gru_9/while/Identity"5
gru_9_while_identitygru_9/while/Identity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
јD
ф
E__inference_lstm_19_layer_call_and_return_conditional_losses_44727160

inputs
lstm_cell_19_44727078
lstm_cell_19_44727080
lstm_cell_19_44727082
identityЂ$lstm_cell_19/StatefulPartitionedCallЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџK*
shrink_axis_mask2
strided_slice_2Ј
$lstm_cell_19/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_19_44727078lstm_cell_19_44727080lstm_cell_19_44727082*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_lstm_cell_19_layer_call_and_return_conditional_losses_447267642&
$lstm_cell_19/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter­
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_19_44727078lstm_cell_19_44727080lstm_cell_19_44727082*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_44727091*
condR
while_cond_44727090*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime
IdentityIdentitystrided_slice_3:output:0%^lstm_cell_19/StatefulPartitionedCall^while*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџK:::2L
$lstm_cell_19/StatefulPartitionedCall$lstm_cell_19/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK
 
_user_specified_nameinputs
Т
Я
/__inference_lstm_cell_19_layer_call_fn_44732179

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2ЂStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_lstm_cell_19_layer_call_and_return_conditional_losses_447267642
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:џџџџџџџџџK:џџџџџџџџџ2:џџџџџџџџџ2:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџK
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ2
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџ2
"
_user_specified_name
states/1
Е
п
J__inference_lstm_cell_18_layer_call_and_return_conditional_losses_44731921

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ќ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	KЌ*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ќ*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimП
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџK2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:џџџџџџџџџK2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџK2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџK2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
mul_2Ј
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2

IdentityЌ

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2

Identity_1Ќ

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:џџџџџџџџџ:џџџџџџџџџK:џџџџџџџџџK:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџK
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџK
"
_user_specified_name
states/1
Ы
f
H__inference_dropout_19_layer_call_and_return_conditional_losses_44728034

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ22

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:џџџџџџџџџ2:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
ф

+__inference_dense_28_layer_call_fn_44731856

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_28_layer_call_and_return_conditional_losses_447284202
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ2::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
џF
Ї
while_body_44730522
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
*while_gru_cell_9_readvariableop_resource_05
1while_gru_cell_9_matmul_readvariableop_resource_07
3while_gru_cell_9_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
(while_gru_cell_9_readvariableop_resource3
/while_gru_cell_9_matmul_readvariableop_resource5
1while_gru_cell_9_matmul_1_readvariableop_resourceЂ&while/gru_cell_9/MatMul/ReadVariableOpЂ(while/gru_cell_9/MatMul_1/ReadVariableOpЂwhile/gru_cell_9/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЎ
while/gru_cell_9/ReadVariableOpReadVariableOp*while_gru_cell_9_readvariableop_resource_0*
_output_shapes
:	*
dtype02!
while/gru_cell_9/ReadVariableOp
while/gru_cell_9/unstackUnpack'while/gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
while/gru_cell_9/unstackУ
&while/gru_cell_9/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/gru_cell_9/MatMul/ReadVariableOpб
while/gru_cell_9/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
while/gru_cell_9/MatMulИ
while/gru_cell_9/BiasAddBiasAdd!while/gru_cell_9/MatMul:product:0!while/gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
while/gru_cell_9/BiasAddr
while/gru_cell_9/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_9/Const
 while/gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2"
 while/gru_cell_9/split/split_dim№
while/gru_cell_9/splitSplit)while/gru_cell_9/split/split_dim:output:0!while/gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/gru_cell_9/splitЩ
(while/gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes
:	2*
dtype02*
(while/gru_cell_9/MatMul_1/ReadVariableOpК
while/gru_cell_9/MatMul_1MatMulwhile_placeholder_20while/gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
while/gru_cell_9/MatMul_1О
while/gru_cell_9/BiasAdd_1BiasAdd#while/gru_cell_9/MatMul_1:product:0!while/gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2
while/gru_cell_9/BiasAdd_1
while/gru_cell_9/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   џџџџ2
while/gru_cell_9/Const_1
"while/gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2$
"while/gru_cell_9/split_1/split_dimЈ
while/gru_cell_9/split_1SplitV#while/gru_cell_9/BiasAdd_1:output:0!while/gru_cell_9/Const_1:output:0+while/gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/gru_cell_9/split_1Ћ
while/gru_cell_9/addAddV2while/gru_cell_9/split:output:0!while/gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/add
while/gru_cell_9/SigmoidSigmoidwhile/gru_cell_9/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/SigmoidЏ
while/gru_cell_9/add_1AddV2while/gru_cell_9/split:output:1!while/gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/add_1
while/gru_cell_9/Sigmoid_1Sigmoidwhile/gru_cell_9/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/Sigmoid_1Ј
while/gru_cell_9/mulMulwhile/gru_cell_9/Sigmoid_1:y:0!while/gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/mulІ
while/gru_cell_9/add_2AddV2while/gru_cell_9/split:output:2while/gru_cell_9/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/add_2
while/gru_cell_9/ReluReluwhile/gru_cell_9/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/Relu
while/gru_cell_9/mul_1Mulwhile/gru_cell_9/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/mul_1u
while/gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_9/sub/xЄ
while/gru_cell_9/subSubwhile/gru_cell_9/sub/x:output:0while/gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/subЈ
while/gru_cell_9/mul_2Mulwhile/gru_cell_9/sub:z:0#while/gru_cell_9/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/mul_2Ѓ
while/gru_cell_9/add_3AddV2while/gru_cell_9/mul_1:z:0while/gru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/add_3о
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_9/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1д
while/IdentityIdentitywhile/add_1:z:0'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityч
while/Identity_1Identitywhile_while_maximum_iterations'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1ж
while/Identity_2Identitywhile/add:z:0'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3є
while/Identity_4Identitywhile/gru_cell_9/add_3:z:0'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4"h
1while_gru_cell_9_matmul_1_readvariableop_resource3while_gru_cell_9_matmul_1_readvariableop_resource_0"d
/while_gru_cell_9_matmul_readvariableop_resource1while_gru_cell_9_matmul_readvariableop_resource_0"V
(while_gru_cell_9_readvariableop_resource*while_gru_cell_9_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ2: : :::2P
&while/gru_cell_9/MatMul/ReadVariableOp&while/gru_cell_9/MatMul/ReadVariableOp2T
(while/gru_cell_9/MatMul_1/ReadVariableOp(while/gru_cell_9/MatMul_1/ReadVariableOp2B
while/gru_cell_9/ReadVariableOpwhile/gru_cell_9/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
Є

Ц
*__inference_model_9_layer_call_fn_44729735

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13
identityЂStatefulPartitionedCallЅ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*1
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_model_9_layer_call_and_return_conditional_losses_447285682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:џџџџџџџџџџџџџџџџџџ:::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
O


lstm_19_while_body_44729112,
(lstm_19_while_lstm_19_while_loop_counter2
.lstm_19_while_lstm_19_while_maximum_iterations
lstm_19_while_placeholder
lstm_19_while_placeholder_1
lstm_19_while_placeholder_2
lstm_19_while_placeholder_3+
'lstm_19_while_lstm_19_strided_slice_1_0g
clstm_19_while_tensorarrayv2read_tensorlistgetitem_lstm_19_tensorarrayunstack_tensorlistfromtensor_0?
;lstm_19_while_lstm_cell_19_matmul_readvariableop_resource_0A
=lstm_19_while_lstm_cell_19_matmul_1_readvariableop_resource_0@
<lstm_19_while_lstm_cell_19_biasadd_readvariableop_resource_0
lstm_19_while_identity
lstm_19_while_identity_1
lstm_19_while_identity_2
lstm_19_while_identity_3
lstm_19_while_identity_4
lstm_19_while_identity_5)
%lstm_19_while_lstm_19_strided_slice_1e
alstm_19_while_tensorarrayv2read_tensorlistgetitem_lstm_19_tensorarrayunstack_tensorlistfromtensor=
9lstm_19_while_lstm_cell_19_matmul_readvariableop_resource?
;lstm_19_while_lstm_cell_19_matmul_1_readvariableop_resource>
:lstm_19_while_lstm_cell_19_biasadd_readvariableop_resourceЂ1lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOpЂ0lstm_19/while/lstm_cell_19/MatMul/ReadVariableOpЂ2lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOpг
?lstm_19/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   2A
?lstm_19/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_19/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_19_while_tensorarrayv2read_tensorlistgetitem_lstm_19_tensorarrayunstack_tensorlistfromtensor_0lstm_19_while_placeholderHlstm_19/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџK*
element_dtype023
1lstm_19/while/TensorArrayV2Read/TensorListGetItemс
0lstm_19/while/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp;lstm_19_while_lstm_cell_19_matmul_readvariableop_resource_0*
_output_shapes
:	KШ*
dtype022
0lstm_19/while/lstm_cell_19/MatMul/ReadVariableOpї
!lstm_19/while/lstm_cell_19/MatMulMatMul8lstm_19/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2#
!lstm_19/while/lstm_cell_19/MatMulч
2lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp=lstm_19_while_lstm_cell_19_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype024
2lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOpр
#lstm_19/while/lstm_cell_19/MatMul_1MatMullstm_19_while_placeholder_2:lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2%
#lstm_19/while/lstm_cell_19/MatMul_1и
lstm_19/while/lstm_cell_19/addAddV2+lstm_19/while/lstm_cell_19/MatMul:product:0-lstm_19/while/lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2 
lstm_19/while/lstm_cell_19/addр
1lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp<lstm_19_while_lstm_cell_19_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype023
1lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOpх
"lstm_19/while/lstm_cell_19/BiasAddBiasAdd"lstm_19/while/lstm_cell_19/add:z:09lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2$
"lstm_19/while/lstm_cell_19/BiasAdd
 lstm_19/while/lstm_cell_19/ConstConst*
_output_shapes
: *
dtype0*
value	B :2"
 lstm_19/while/lstm_cell_19/Const
*lstm_19/while/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_19/while/lstm_cell_19/split/split_dimЋ
 lstm_19/while/lstm_cell_19/splitSplit3lstm_19/while/lstm_cell_19/split/split_dim:output:0+lstm_19/while/lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2"
 lstm_19/while/lstm_cell_19/splitА
"lstm_19/while/lstm_cell_19/SigmoidSigmoid)lstm_19/while/lstm_cell_19/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"lstm_19/while/lstm_cell_19/SigmoidД
$lstm_19/while/lstm_cell_19/Sigmoid_1Sigmoid)lstm_19/while/lstm_cell_19/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22&
$lstm_19/while/lstm_cell_19/Sigmoid_1Р
lstm_19/while/lstm_cell_19/mulMul(lstm_19/while/lstm_cell_19/Sigmoid_1:y:0lstm_19_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22 
lstm_19/while/lstm_cell_19/mulЇ
lstm_19/while/lstm_cell_19/ReluRelu)lstm_19/while/lstm_cell_19/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22!
lstm_19/while/lstm_cell_19/Reluд
 lstm_19/while/lstm_cell_19/mul_1Mul&lstm_19/while/lstm_cell_19/Sigmoid:y:0-lstm_19/while/lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22"
 lstm_19/while/lstm_cell_19/mul_1Щ
 lstm_19/while/lstm_cell_19/add_1AddV2"lstm_19/while/lstm_cell_19/mul:z:0$lstm_19/while/lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22"
 lstm_19/while/lstm_cell_19/add_1Д
$lstm_19/while/lstm_cell_19/Sigmoid_2Sigmoid)lstm_19/while/lstm_cell_19/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22&
$lstm_19/while/lstm_cell_19/Sigmoid_2І
!lstm_19/while/lstm_cell_19/Relu_1Relu$lstm_19/while/lstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22#
!lstm_19/while/lstm_cell_19/Relu_1и
 lstm_19/while/lstm_cell_19/mul_2Mul(lstm_19/while/lstm_cell_19/Sigmoid_2:y:0/lstm_19/while/lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22"
 lstm_19/while/lstm_cell_19/mul_2
2lstm_19/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_19_while_placeholder_1lstm_19_while_placeholder$lstm_19/while/lstm_cell_19/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_19/while/TensorArrayV2Write/TensorListSetIteml
lstm_19/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_19/while/add/y
lstm_19/while/addAddV2lstm_19_while_placeholderlstm_19/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_19/while/addp
lstm_19/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_19/while/add_1/y
lstm_19/while/add_1AddV2(lstm_19_while_lstm_19_while_loop_counterlstm_19/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_19/while/add_1
lstm_19/while/IdentityIdentitylstm_19/while/add_1:z:02^lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOp1^lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp3^lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_19/while/Identity­
lstm_19/while/Identity_1Identity.lstm_19_while_lstm_19_while_maximum_iterations2^lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOp1^lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp3^lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_19/while/Identity_1
lstm_19/while/Identity_2Identitylstm_19/while/add:z:02^lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOp1^lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp3^lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_19/while/Identity_2С
lstm_19/while/Identity_3IdentityBlstm_19/while/TensorArrayV2Write/TensorListSetItem:output_handle:02^lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOp1^lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp3^lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_19/while/Identity_3Д
lstm_19/while/Identity_4Identity$lstm_19/while/lstm_cell_19/mul_2:z:02^lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOp1^lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp3^lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_19/while/Identity_4Д
lstm_19/while/Identity_5Identity$lstm_19/while/lstm_cell_19/add_1:z:02^lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOp1^lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp3^lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_19/while/Identity_5"9
lstm_19_while_identitylstm_19/while/Identity:output:0"=
lstm_19_while_identity_1!lstm_19/while/Identity_1:output:0"=
lstm_19_while_identity_2!lstm_19/while/Identity_2:output:0"=
lstm_19_while_identity_3!lstm_19/while/Identity_3:output:0"=
lstm_19_while_identity_4!lstm_19/while/Identity_4:output:0"=
lstm_19_while_identity_5!lstm_19/while/Identity_5:output:0"P
%lstm_19_while_lstm_19_strided_slice_1'lstm_19_while_lstm_19_strided_slice_1_0"z
:lstm_19_while_lstm_cell_19_biasadd_readvariableop_resource<lstm_19_while_lstm_cell_19_biasadd_readvariableop_resource_0"|
;lstm_19_while_lstm_cell_19_matmul_1_readvariableop_resource=lstm_19_while_lstm_cell_19_matmul_1_readvariableop_resource_0"x
9lstm_19_while_lstm_cell_19_matmul_readvariableop_resource;lstm_19_while_lstm_cell_19_matmul_readvariableop_resource_0"Ш
alstm_19_while_tensorarrayv2read_tensorlistgetitem_lstm_19_tensorarrayunstack_tensorlistfromtensorclstm_19_while_tensorarrayv2read_tensorlistgetitem_lstm_19_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : :::2f
1lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOp1lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOp2d
0lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp0lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp2h
2lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOp2lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
ќD
ф
E__inference_lstm_18_layer_call_and_return_conditional_losses_44725988

inputs
lstm_cell_18_44725906
lstm_cell_18_44725908
lstm_cell_18_44725910
identityЂ$lstm_cell_18/StatefulPartitionedCallЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2Ј
$lstm_cell_18/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_18_44725906lstm_cell_18_44725908lstm_cell_18_44725910*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_lstm_cell_18_layer_call_and_return_conditional_losses_447255922&
$lstm_cell_18/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter­
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_18_44725906lstm_cell_18_44725908lstm_cell_18_44725910*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџK:џџџџџџџџџK: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_44725919*
condR
while_cond_44725918*K
output_shapes:
8: : : : :џџџџџџџџџK:џџџџџџџџџK: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџK*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime
IdentityIdentitytranspose_1:y:0%^lstm_cell_18/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2L
$lstm_cell_18/StatefulPartitionedCall$lstm_cell_18/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

g
H__inference_dropout_19_layer_call_and_return_conditional_losses_44728029

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeД
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ22
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ2:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
C

while_body_44730166
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_18_matmul_readvariableop_resource_09
5while_lstm_cell_18_matmul_1_readvariableop_resource_08
4while_lstm_cell_18_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_18_matmul_readvariableop_resource7
3while_lstm_cell_18_matmul_1_readvariableop_resource6
2while_lstm_cell_18_biasadd_readvariableop_resourceЂ)while/lstm_cell_18/BiasAdd/ReadVariableOpЂ(while/lstm_cell_18/MatMul/ReadVariableOpЂ*while/lstm_cell_18/MatMul_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЩ
(while/lstm_cell_18/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_18_matmul_readvariableop_resource_0*
_output_shapes
:	Ќ*
dtype02*
(while/lstm_cell_18/MatMul/ReadVariableOpз
while/lstm_cell_18/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_18/MatMulЯ
*while/lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_18_matmul_1_readvariableop_resource_0*
_output_shapes
:	KЌ*
dtype02,
*while/lstm_cell_18/MatMul_1/ReadVariableOpР
while/lstm_cell_18/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_18/MatMul_1И
while/lstm_cell_18/addAddV2#while/lstm_cell_18/MatMul:product:0%while/lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_18/addШ
)while/lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_18_biasadd_readvariableop_resource_0*
_output_shapes	
:Ќ*
dtype02+
)while/lstm_cell_18/BiasAdd/ReadVariableOpХ
while/lstm_cell_18/BiasAddBiasAddwhile/lstm_cell_18/add:z:01while/lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_18/BiasAddv
while/lstm_cell_18/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_18/Const
"while/lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_18/split/split_dim
while/lstm_cell_18/splitSplit+while/lstm_cell_18/split/split_dim:output:0#while/lstm_cell_18/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*
	num_split2
while/lstm_cell_18/split
while/lstm_cell_18/SigmoidSigmoid!while/lstm_cell_18/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/Sigmoid
while/lstm_cell_18/Sigmoid_1Sigmoid!while/lstm_cell_18/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/Sigmoid_1 
while/lstm_cell_18/mulMul while/lstm_cell_18/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/mul
while/lstm_cell_18/ReluRelu!while/lstm_cell_18/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/ReluД
while/lstm_cell_18/mul_1Mulwhile/lstm_cell_18/Sigmoid:y:0%while/lstm_cell_18/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/mul_1Љ
while/lstm_cell_18/add_1AddV2while/lstm_cell_18/mul:z:0while/lstm_cell_18/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/add_1
while/lstm_cell_18/Sigmoid_2Sigmoid!while/lstm_cell_18/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/Sigmoid_2
while/lstm_cell_18/Relu_1Reluwhile/lstm_cell_18/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/Relu_1И
while/lstm_cell_18/mul_2Mul while/lstm_cell_18/Sigmoid_2:y:0'while/lstm_cell_18/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_18/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1т
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_18/BiasAdd/ReadVariableOp)^while/lstm_cell_18/MatMul/ReadVariableOp+^while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityѕ
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_18/BiasAdd/ReadVariableOp)^while/lstm_cell_18/MatMul/ReadVariableOp+^while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1ф
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_18/BiasAdd/ReadVariableOp)^while/lstm_cell_18/MatMul/ReadVariableOp+^while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_18/BiasAdd/ReadVariableOp)^while/lstm_cell_18/MatMul/ReadVariableOp+^while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_18/mul_2:z:0*^while/lstm_cell_18/BiasAdd/ReadVariableOp)^while/lstm_cell_18/MatMul/ReadVariableOp+^while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_18/add_1:z:0*^while/lstm_cell_18/BiasAdd/ReadVariableOp)^while/lstm_cell_18/MatMul/ReadVariableOp+^while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_18_biasadd_readvariableop_resource4while_lstm_cell_18_biasadd_readvariableop_resource_0"l
3while_lstm_cell_18_matmul_1_readvariableop_resource5while_lstm_cell_18_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_18_matmul_readvariableop_resource3while_lstm_cell_18_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџK:џџџџџџџџџK: : :::2V
)while/lstm_cell_18/BiasAdd/ReadVariableOp)while/lstm_cell_18/BiasAdd/ReadVariableOp2T
(while/lstm_cell_18/MatMul/ReadVariableOp(while/lstm_cell_18/MatMul/ReadVariableOp2X
*while/lstm_cell_18/MatMul_1/ReadVariableOp*while/lstm_cell_18/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџK:-)
'
_output_shapes
:џџџџџџџџџK:

_output_shapes
: :

_output_shapes
: 
Ы
­
H__inference_gru_cell_9_layer_call_and_return_conditional_losses_44726201

inputs

states
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1ЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split/split_dimЌ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   џџџџ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split_1/split_dimг
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
Relu\
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:џџџџџџџџџ22
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
subd
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
add_3 
IdentityIdentity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22

IdentityЄ

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ2:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_namestates
C

while_body_44731529
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_19_matmul_readvariableop_resource_09
5while_lstm_cell_19_matmul_1_readvariableop_resource_08
4while_lstm_cell_19_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_19_matmul_readvariableop_resource7
3while_lstm_cell_19_matmul_1_readvariableop_resource6
2while_lstm_cell_19_biasadd_readvariableop_resourceЂ)while/lstm_cell_19/BiasAdd/ReadVariableOpЂ(while/lstm_cell_19/MatMul/ReadVariableOpЂ*while/lstm_cell_19/MatMul_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџK*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЩ
(while/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_19_matmul_readvariableop_resource_0*
_output_shapes
:	KШ*
dtype02*
(while/lstm_cell_19/MatMul/ReadVariableOpз
while/lstm_cell_19/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_19/MatMulЯ
*while/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_19_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02,
*while/lstm_cell_19/MatMul_1/ReadVariableOpР
while/lstm_cell_19/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_19/MatMul_1И
while/lstm_cell_19/addAddV2#while/lstm_cell_19/MatMul:product:0%while/lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_19/addШ
)while/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_19_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02+
)while/lstm_cell_19/BiasAdd/ReadVariableOpХ
while/lstm_cell_19/BiasAddBiasAddwhile/lstm_cell_19/add:z:01while/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_19/BiasAddv
while/lstm_cell_19/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_19/Const
"while/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_19/split/split_dim
while/lstm_cell_19/splitSplit+while/lstm_cell_19/split/split_dim:output:0#while/lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_19/split
while/lstm_cell_19/SigmoidSigmoid!while/lstm_cell_19/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/Sigmoid
while/lstm_cell_19/Sigmoid_1Sigmoid!while/lstm_cell_19/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/Sigmoid_1 
while/lstm_cell_19/mulMul while/lstm_cell_19/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/mul
while/lstm_cell_19/ReluRelu!while/lstm_cell_19/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/ReluД
while/lstm_cell_19/mul_1Mulwhile/lstm_cell_19/Sigmoid:y:0%while/lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/mul_1Љ
while/lstm_cell_19/add_1AddV2while/lstm_cell_19/mul:z:0while/lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/add_1
while/lstm_cell_19/Sigmoid_2Sigmoid!while/lstm_cell_19/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/Sigmoid_2
while/lstm_cell_19/Relu_1Reluwhile/lstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/Relu_1И
while/lstm_cell_19/mul_2Mul while/lstm_cell_19/Sigmoid_2:y:0'while/lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_19/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1т
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityѕ
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1ф
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_19/mul_2:z:0*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_19/add_1:z:0*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_19_biasadd_readvariableop_resource4while_lstm_cell_19_biasadd_readvariableop_resource_0"l
3while_lstm_cell_19_matmul_1_readvariableop_resource5while_lstm_cell_19_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_19_matmul_readvariableop_resource3while_lstm_cell_19_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : :::2V
)while/lstm_cell_19/BiasAdd/ReadVariableOp)while/lstm_cell_19/BiasAdd/ReadVariableOp2T
(while/lstm_cell_19/MatMul/ReadVariableOp(while/lstm_cell_19/MatMul/ReadVariableOp2X
*while/lstm_cell_19/MatMul_1/ReadVariableOp*while/lstm_cell_19/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
Е
Э
while_cond_44727371
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_44727371___redundant_placeholder06
2while_while_cond_44727371___redundant_placeholder16
2while_while_cond_44727371___redundant_placeholder26
2while_while_cond_44727371___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџK:џџџџџџџџџK: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџK:-)
'
_output_shapes
:џџџџџџџџџK:

_output_shapes
: :

_output_shapes
:
џF
Ї
while_body_44730681
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
*while_gru_cell_9_readvariableop_resource_05
1while_gru_cell_9_matmul_readvariableop_resource_07
3while_gru_cell_9_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
(while_gru_cell_9_readvariableop_resource3
/while_gru_cell_9_matmul_readvariableop_resource5
1while_gru_cell_9_matmul_1_readvariableop_resourceЂ&while/gru_cell_9/MatMul/ReadVariableOpЂ(while/gru_cell_9/MatMul_1/ReadVariableOpЂwhile/gru_cell_9/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЎ
while/gru_cell_9/ReadVariableOpReadVariableOp*while_gru_cell_9_readvariableop_resource_0*
_output_shapes
:	*
dtype02!
while/gru_cell_9/ReadVariableOp
while/gru_cell_9/unstackUnpack'while/gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
while/gru_cell_9/unstackУ
&while/gru_cell_9/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/gru_cell_9/MatMul/ReadVariableOpб
while/gru_cell_9/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
while/gru_cell_9/MatMulИ
while/gru_cell_9/BiasAddBiasAdd!while/gru_cell_9/MatMul:product:0!while/gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
while/gru_cell_9/BiasAddr
while/gru_cell_9/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_9/Const
 while/gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2"
 while/gru_cell_9/split/split_dim№
while/gru_cell_9/splitSplit)while/gru_cell_9/split/split_dim:output:0!while/gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/gru_cell_9/splitЩ
(while/gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes
:	2*
dtype02*
(while/gru_cell_9/MatMul_1/ReadVariableOpК
while/gru_cell_9/MatMul_1MatMulwhile_placeholder_20while/gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
while/gru_cell_9/MatMul_1О
while/gru_cell_9/BiasAdd_1BiasAdd#while/gru_cell_9/MatMul_1:product:0!while/gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2
while/gru_cell_9/BiasAdd_1
while/gru_cell_9/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   џџџџ2
while/gru_cell_9/Const_1
"while/gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2$
"while/gru_cell_9/split_1/split_dimЈ
while/gru_cell_9/split_1SplitV#while/gru_cell_9/BiasAdd_1:output:0!while/gru_cell_9/Const_1:output:0+while/gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/gru_cell_9/split_1Ћ
while/gru_cell_9/addAddV2while/gru_cell_9/split:output:0!while/gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/add
while/gru_cell_9/SigmoidSigmoidwhile/gru_cell_9/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/SigmoidЏ
while/gru_cell_9/add_1AddV2while/gru_cell_9/split:output:1!while/gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/add_1
while/gru_cell_9/Sigmoid_1Sigmoidwhile/gru_cell_9/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/Sigmoid_1Ј
while/gru_cell_9/mulMulwhile/gru_cell_9/Sigmoid_1:y:0!while/gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/mulІ
while/gru_cell_9/add_2AddV2while/gru_cell_9/split:output:2while/gru_cell_9/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/add_2
while/gru_cell_9/ReluReluwhile/gru_cell_9/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/Relu
while/gru_cell_9/mul_1Mulwhile/gru_cell_9/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/mul_1u
while/gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_9/sub/xЄ
while/gru_cell_9/subSubwhile/gru_cell_9/sub/x:output:0while/gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/subЈ
while/gru_cell_9/mul_2Mulwhile/gru_cell_9/sub:z:0#while/gru_cell_9/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/mul_2Ѓ
while/gru_cell_9/add_3AddV2while/gru_cell_9/mul_1:z:0while/gru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/add_3о
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_9/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1д
while/IdentityIdentitywhile/add_1:z:0'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityч
while/Identity_1Identitywhile_while_maximum_iterations'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1ж
while/Identity_2Identitywhile/add:z:0'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3є
while/Identity_4Identitywhile/gru_cell_9/add_3:z:0'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4"h
1while_gru_cell_9_matmul_1_readvariableop_resource3while_gru_cell_9_matmul_1_readvariableop_resource_0"d
/while_gru_cell_9_matmul_readvariableop_resource1while_gru_cell_9_matmul_readvariableop_resource_0"V
(while_gru_cell_9_readvariableop_resource*while_gru_cell_9_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ2: : :::2P
&while/gru_cell_9/MatMul/ReadVariableOp&while/gru_cell_9/MatMul/ReadVariableOp2T
(while/gru_cell_9/MatMul_1/ReadVariableOp(while/gru_cell_9/MatMul_1/ReadVariableOp2B
while/gru_cell_9/ReadVariableOpwhile/gru_cell_9/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
C

while_body_44731201
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_19_matmul_readvariableop_resource_09
5while_lstm_cell_19_matmul_1_readvariableop_resource_08
4while_lstm_cell_19_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_19_matmul_readvariableop_resource7
3while_lstm_cell_19_matmul_1_readvariableop_resource6
2while_lstm_cell_19_biasadd_readvariableop_resourceЂ)while/lstm_cell_19/BiasAdd/ReadVariableOpЂ(while/lstm_cell_19/MatMul/ReadVariableOpЂ*while/lstm_cell_19/MatMul_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџK*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЩ
(while/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_19_matmul_readvariableop_resource_0*
_output_shapes
:	KШ*
dtype02*
(while/lstm_cell_19/MatMul/ReadVariableOpз
while/lstm_cell_19/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_19/MatMulЯ
*while/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_19_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02,
*while/lstm_cell_19/MatMul_1/ReadVariableOpР
while/lstm_cell_19/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_19/MatMul_1И
while/lstm_cell_19/addAddV2#while/lstm_cell_19/MatMul:product:0%while/lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_19/addШ
)while/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_19_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02+
)while/lstm_cell_19/BiasAdd/ReadVariableOpХ
while/lstm_cell_19/BiasAddBiasAddwhile/lstm_cell_19/add:z:01while/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_19/BiasAddv
while/lstm_cell_19/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_19/Const
"while/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_19/split/split_dim
while/lstm_cell_19/splitSplit+while/lstm_cell_19/split/split_dim:output:0#while/lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_19/split
while/lstm_cell_19/SigmoidSigmoid!while/lstm_cell_19/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/Sigmoid
while/lstm_cell_19/Sigmoid_1Sigmoid!while/lstm_cell_19/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/Sigmoid_1 
while/lstm_cell_19/mulMul while/lstm_cell_19/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/mul
while/lstm_cell_19/ReluRelu!while/lstm_cell_19/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/ReluД
while/lstm_cell_19/mul_1Mulwhile/lstm_cell_19/Sigmoid:y:0%while/lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/mul_1Љ
while/lstm_cell_19/add_1AddV2while/lstm_cell_19/mul:z:0while/lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/add_1
while/lstm_cell_19/Sigmoid_2Sigmoid!while/lstm_cell_19/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/Sigmoid_2
while/lstm_cell_19/Relu_1Reluwhile/lstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/Relu_1И
while/lstm_cell_19/mul_2Mul while/lstm_cell_19/Sigmoid_2:y:0'while/lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_19/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1т
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityѕ
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1ф
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_19/mul_2:z:0*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_19/add_1:z:0*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_19_biasadd_readvariableop_resource4while_lstm_cell_19_biasadd_readvariableop_resource_0"l
3while_lstm_cell_19_matmul_1_readvariableop_resource5while_lstm_cell_19_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_19_matmul_readvariableop_resource3while_lstm_cell_19_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : :::2V
)while/lstm_cell_19/BiasAdd/ReadVariableOp)while/lstm_cell_19/BiasAdd/ReadVariableOp2T
(while/lstm_cell_19/MatMul/ReadVariableOp(while/lstm_cell_19/MatMul/ReadVariableOp2X
*while/lstm_cell_19/MatMul_1/ReadVariableOp*while/lstm_cell_19/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
џ
f
H__inference_dropout_18_layer_call_and_return_conditional_losses_44730443

inputs

identity_1g
IdentityIdentityinputs*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2

Identityv

Identity_1IdentityIdentity:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2

Identity_1"!

identity_1Identity_1:output:0*3
_input_shapes"
 :џџџџџџџџџџџџџџџџџџK:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK
 
_user_specified_nameinputs
г
Џ
H__inference_gru_cell_9_layer_call_and_return_conditional_losses_44732068

inputs
states_0
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1ЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split/split_dimЌ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   џџџџ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split_1/split_dimг
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
Relu^
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:џџџџџџџџџ22
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
subd
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
add_3 
IdentityIdentity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22

IdentityЄ

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ2:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ2
"
_user_specified_name
states/0
Е
Э
while_cond_44727090
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_44727090___redundant_placeholder06
2while_while_cond_44727090___redundant_placeholder16
2while_while_cond_44727090___redundant_placeholder26
2while_while_cond_44727090___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџ2:џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
Е
Э
while_cond_44731353
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_44731353___redundant_placeholder06
2while_while_cond_44731353___redundant_placeholder16
2while_while_cond_44731353___redundant_placeholder26
2while_while_cond_44731353___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџ2:џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
	
п
F__inference_dense_29_layer_call_and_return_conditional_losses_44728462

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ`::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ`
 
_user_specified_nameinputs
к
Д
while_cond_44727866
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_44727866___redundant_placeholder06
2while_while_cond_44727866___redundant_placeholder16
2while_while_cond_44727866___redundant_placeholder26
2while_while_cond_44727866___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
аP
л
gru_9_while_body_44729437(
$gru_9_while_gru_9_while_loop_counter.
*gru_9_while_gru_9_while_maximum_iterations
gru_9_while_placeholder
gru_9_while_placeholder_1
gru_9_while_placeholder_2'
#gru_9_while_gru_9_strided_slice_1_0c
_gru_9_while_tensorarrayv2read_tensorlistgetitem_gru_9_tensorarrayunstack_tensorlistfromtensor_04
0gru_9_while_gru_cell_9_readvariableop_resource_0;
7gru_9_while_gru_cell_9_matmul_readvariableop_resource_0=
9gru_9_while_gru_cell_9_matmul_1_readvariableop_resource_0
gru_9_while_identity
gru_9_while_identity_1
gru_9_while_identity_2
gru_9_while_identity_3
gru_9_while_identity_4%
!gru_9_while_gru_9_strided_slice_1a
]gru_9_while_tensorarrayv2read_tensorlistgetitem_gru_9_tensorarrayunstack_tensorlistfromtensor2
.gru_9_while_gru_cell_9_readvariableop_resource9
5gru_9_while_gru_cell_9_matmul_readvariableop_resource;
7gru_9_while_gru_cell_9_matmul_1_readvariableop_resourceЂ,gru_9/while/gru_cell_9/MatMul/ReadVariableOpЂ.gru_9/while/gru_cell_9/MatMul_1/ReadVariableOpЂ%gru_9/while/gru_cell_9/ReadVariableOpЯ
=gru_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2?
=gru_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeї
/gru_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_9_while_tensorarrayv2read_tensorlistgetitem_gru_9_tensorarrayunstack_tensorlistfromtensor_0gru_9_while_placeholderFgru_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype021
/gru_9/while/TensorArrayV2Read/TensorListGetItemР
%gru_9/while/gru_cell_9/ReadVariableOpReadVariableOp0gru_9_while_gru_cell_9_readvariableop_resource_0*
_output_shapes
:	*
dtype02'
%gru_9/while/gru_cell_9/ReadVariableOpБ
gru_9/while/gru_cell_9/unstackUnpack-gru_9/while/gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2 
gru_9/while/gru_cell_9/unstackе
,gru_9/while/gru_cell_9/MatMul/ReadVariableOpReadVariableOp7gru_9_while_gru_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02.
,gru_9/while/gru_cell_9/MatMul/ReadVariableOpщ
gru_9/while/gru_cell_9/MatMulMatMul6gru_9/while/TensorArrayV2Read/TensorListGetItem:item:04gru_9/while/gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_9/while/gru_cell_9/MatMulа
gru_9/while/gru_cell_9/BiasAddBiasAdd'gru_9/while/gru_cell_9/MatMul:product:0'gru_9/while/gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2 
gru_9/while/gru_cell_9/BiasAdd~
gru_9/while/gru_cell_9/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_9/while/gru_cell_9/Const
&gru_9/while/gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2(
&gru_9/while/gru_cell_9/split/split_dim
gru_9/while/gru_cell_9/splitSplit/gru_9/while/gru_cell_9/split/split_dim:output:0'gru_9/while/gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
gru_9/while/gru_cell_9/splitл
.gru_9/while/gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp9gru_9_while_gru_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes
:	2*
dtype020
.gru_9/while/gru_cell_9/MatMul_1/ReadVariableOpв
gru_9/while/gru_cell_9/MatMul_1MatMulgru_9_while_placeholder_26gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2!
gru_9/while/gru_cell_9/MatMul_1ж
 gru_9/while/gru_cell_9/BiasAdd_1BiasAdd)gru_9/while/gru_cell_9/MatMul_1:product:0'gru_9/while/gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2"
 gru_9/while/gru_cell_9/BiasAdd_1
gru_9/while/gru_cell_9/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   џџџџ2 
gru_9/while/gru_cell_9/Const_1
(gru_9/while/gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2*
(gru_9/while/gru_cell_9/split_1/split_dimЦ
gru_9/while/gru_cell_9/split_1SplitV)gru_9/while/gru_cell_9/BiasAdd_1:output:0'gru_9/while/gru_cell_9/Const_1:output:01gru_9/while/gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2 
gru_9/while/gru_cell_9/split_1У
gru_9/while/gru_cell_9/addAddV2%gru_9/while/gru_cell_9/split:output:0'gru_9/while/gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_9/while/gru_cell_9/add
gru_9/while/gru_cell_9/SigmoidSigmoidgru_9/while/gru_cell_9/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22 
gru_9/while/gru_cell_9/SigmoidЧ
gru_9/while/gru_cell_9/add_1AddV2%gru_9/while/gru_cell_9/split:output:1'gru_9/while/gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_9/while/gru_cell_9/add_1Ѓ
 gru_9/while/gru_cell_9/Sigmoid_1Sigmoid gru_9/while/gru_cell_9/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22"
 gru_9/while/gru_cell_9/Sigmoid_1Р
gru_9/while/gru_cell_9/mulMul$gru_9/while/gru_cell_9/Sigmoid_1:y:0'gru_9/while/gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_9/while/gru_cell_9/mulО
gru_9/while/gru_cell_9/add_2AddV2%gru_9/while/gru_cell_9/split:output:2gru_9/while/gru_cell_9/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_9/while/gru_cell_9/add_2
gru_9/while/gru_cell_9/ReluRelu gru_9/while/gru_cell_9/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_9/while/gru_cell_9/ReluД
gru_9/while/gru_cell_9/mul_1Mul"gru_9/while/gru_cell_9/Sigmoid:y:0gru_9_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_9/while/gru_cell_9/mul_1
gru_9/while/gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_9/while/gru_cell_9/sub/xМ
gru_9/while/gru_cell_9/subSub%gru_9/while/gru_cell_9/sub/x:output:0"gru_9/while/gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_9/while/gru_cell_9/subР
gru_9/while/gru_cell_9/mul_2Mulgru_9/while/gru_cell_9/sub:z:0)gru_9/while/gru_cell_9/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_9/while/gru_cell_9/mul_2Л
gru_9/while/gru_cell_9/add_3AddV2 gru_9/while/gru_cell_9/mul_1:z:0 gru_9/while/gru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_9/while/gru_cell_9/add_3ќ
0gru_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_9_while_placeholder_1gru_9_while_placeholder gru_9/while/gru_cell_9/add_3:z:0*
_output_shapes
: *
element_dtype022
0gru_9/while/TensorArrayV2Write/TensorListSetItemh
gru_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_9/while/add/y
gru_9/while/addAddV2gru_9_while_placeholdergru_9/while/add/y:output:0*
T0*
_output_shapes
: 2
gru_9/while/addl
gru_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_9/while/add_1/y
gru_9/while/add_1AddV2$gru_9_while_gru_9_while_loop_countergru_9/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru_9/while/add_1ј
gru_9/while/IdentityIdentitygru_9/while/add_1:z:0-^gru_9/while/gru_cell_9/MatMul/ReadVariableOp/^gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp&^gru_9/while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
gru_9/while/Identity
gru_9/while/Identity_1Identity*gru_9_while_gru_9_while_maximum_iterations-^gru_9/while/gru_cell_9/MatMul/ReadVariableOp/^gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp&^gru_9/while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
gru_9/while/Identity_1њ
gru_9/while/Identity_2Identitygru_9/while/add:z:0-^gru_9/while/gru_cell_9/MatMul/ReadVariableOp/^gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp&^gru_9/while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
gru_9/while/Identity_2Ї
gru_9/while/Identity_3Identity@gru_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0-^gru_9/while/gru_cell_9/MatMul/ReadVariableOp/^gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp&^gru_9/while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
gru_9/while/Identity_3
gru_9/while/Identity_4Identity gru_9/while/gru_cell_9/add_3:z:0-^gru_9/while/gru_cell_9/MatMul/ReadVariableOp/^gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp&^gru_9/while/gru_cell_9/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_9/while/Identity_4"H
!gru_9_while_gru_9_strided_slice_1#gru_9_while_gru_9_strided_slice_1_0"t
7gru_9_while_gru_cell_9_matmul_1_readvariableop_resource9gru_9_while_gru_cell_9_matmul_1_readvariableop_resource_0"p
5gru_9_while_gru_cell_9_matmul_readvariableop_resource7gru_9_while_gru_cell_9_matmul_readvariableop_resource_0"b
.gru_9_while_gru_cell_9_readvariableop_resource0gru_9_while_gru_cell_9_readvariableop_resource_0"5
gru_9_while_identitygru_9/while/Identity:output:0"9
gru_9_while_identity_1gru_9/while/Identity_1:output:0"9
gru_9_while_identity_2gru_9/while/Identity_2:output:0"9
gru_9_while_identity_3gru_9/while/Identity_3:output:0"9
gru_9_while_identity_4gru_9/while/Identity_4:output:0"Р
]gru_9_while_tensorarrayv2read_tensorlistgetitem_gru_9_tensorarrayunstack_tensorlistfromtensor_gru_9_while_tensorarrayv2read_tensorlistgetitem_gru_9_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ2: : :::2\
,gru_9/while/gru_cell_9/MatMul/ReadVariableOp,gru_9/while/gru_cell_9/MatMul/ReadVariableOp2`
.gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp.gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp2N
%gru_9/while/gru_cell_9/ReadVariableOp%gru_9/while/gru_cell_9/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
Е
Э
while_cond_44729990
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_44729990___redundant_placeholder06
2while_while_cond_44729990___redundant_placeholder16
2while_while_cond_44729990___redundant_placeholder26
2while_while_cond_44729990___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџK:џџџџџџџџџK: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџK:-)
'
_output_shapes
:џџџџџџџџџK:

_output_shapes
: :

_output_shapes
:
№	
п
F__inference_dense_28_layer_call_and_return_conditional_losses_44728420

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
М

*__inference_lstm_18_layer_call_fn_44730415
inputs_0
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_lstm_18_layer_call_and_return_conditional_losses_447259882
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
н
f
-__inference_dropout_18_layer_call_fn_44730448

inputs
identityЂStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dropout_18_layer_call_and_return_conditional_losses_447279992
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :џџџџџџџџџџџџџџџџџџK22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK
 
_user_specified_nameinputs
М

*__inference_lstm_18_layer_call_fn_44730426
inputs_0
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_lstm_18_layer_call_and_return_conditional_losses_447261202
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
O


lstm_18_while_body_44728792,
(lstm_18_while_lstm_18_while_loop_counter2
.lstm_18_while_lstm_18_while_maximum_iterations
lstm_18_while_placeholder
lstm_18_while_placeholder_1
lstm_18_while_placeholder_2
lstm_18_while_placeholder_3+
'lstm_18_while_lstm_18_strided_slice_1_0g
clstm_18_while_tensorarrayv2read_tensorlistgetitem_lstm_18_tensorarrayunstack_tensorlistfromtensor_0?
;lstm_18_while_lstm_cell_18_matmul_readvariableop_resource_0A
=lstm_18_while_lstm_cell_18_matmul_1_readvariableop_resource_0@
<lstm_18_while_lstm_cell_18_biasadd_readvariableop_resource_0
lstm_18_while_identity
lstm_18_while_identity_1
lstm_18_while_identity_2
lstm_18_while_identity_3
lstm_18_while_identity_4
lstm_18_while_identity_5)
%lstm_18_while_lstm_18_strided_slice_1e
alstm_18_while_tensorarrayv2read_tensorlistgetitem_lstm_18_tensorarrayunstack_tensorlistfromtensor=
9lstm_18_while_lstm_cell_18_matmul_readvariableop_resource?
;lstm_18_while_lstm_cell_18_matmul_1_readvariableop_resource>
:lstm_18_while_lstm_cell_18_biasadd_readvariableop_resourceЂ1lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOpЂ0lstm_18/while/lstm_cell_18/MatMul/ReadVariableOpЂ2lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOpг
?lstm_18/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2A
?lstm_18/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_18/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_18_while_tensorarrayv2read_tensorlistgetitem_lstm_18_tensorarrayunstack_tensorlistfromtensor_0lstm_18_while_placeholderHlstm_18/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype023
1lstm_18/while/TensorArrayV2Read/TensorListGetItemс
0lstm_18/while/lstm_cell_18/MatMul/ReadVariableOpReadVariableOp;lstm_18_while_lstm_cell_18_matmul_readvariableop_resource_0*
_output_shapes
:	Ќ*
dtype022
0lstm_18/while/lstm_cell_18/MatMul/ReadVariableOpї
!lstm_18/while/lstm_cell_18/MatMulMatMul8lstm_18/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2#
!lstm_18/while/lstm_cell_18/MatMulч
2lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp=lstm_18_while_lstm_cell_18_matmul_1_readvariableop_resource_0*
_output_shapes
:	KЌ*
dtype024
2lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOpр
#lstm_18/while/lstm_cell_18/MatMul_1MatMullstm_18_while_placeholder_2:lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2%
#lstm_18/while/lstm_cell_18/MatMul_1и
lstm_18/while/lstm_cell_18/addAddV2+lstm_18/while/lstm_cell_18/MatMul:product:0-lstm_18/while/lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2 
lstm_18/while/lstm_cell_18/addр
1lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp<lstm_18_while_lstm_cell_18_biasadd_readvariableop_resource_0*
_output_shapes	
:Ќ*
dtype023
1lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOpх
"lstm_18/while/lstm_cell_18/BiasAddBiasAdd"lstm_18/while/lstm_cell_18/add:z:09lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2$
"lstm_18/while/lstm_cell_18/BiasAdd
 lstm_18/while/lstm_cell_18/ConstConst*
_output_shapes
: *
dtype0*
value	B :2"
 lstm_18/while/lstm_cell_18/Const
*lstm_18/while/lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_18/while/lstm_cell_18/split/split_dimЋ
 lstm_18/while/lstm_cell_18/splitSplit3lstm_18/while/lstm_cell_18/split/split_dim:output:0+lstm_18/while/lstm_cell_18/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*
	num_split2"
 lstm_18/while/lstm_cell_18/splitА
"lstm_18/while/lstm_cell_18/SigmoidSigmoid)lstm_18/while/lstm_cell_18/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2$
"lstm_18/while/lstm_cell_18/SigmoidД
$lstm_18/while/lstm_cell_18/Sigmoid_1Sigmoid)lstm_18/while/lstm_cell_18/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџK2&
$lstm_18/while/lstm_cell_18/Sigmoid_1Р
lstm_18/while/lstm_cell_18/mulMul(lstm_18/while/lstm_cell_18/Sigmoid_1:y:0lstm_18_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџK2 
lstm_18/while/lstm_cell_18/mulЇ
lstm_18/while/lstm_cell_18/ReluRelu)lstm_18/while/lstm_cell_18/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџK2!
lstm_18/while/lstm_cell_18/Reluд
 lstm_18/while/lstm_cell_18/mul_1Mul&lstm_18/while/lstm_cell_18/Sigmoid:y:0-lstm_18/while/lstm_cell_18/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2"
 lstm_18/while/lstm_cell_18/mul_1Щ
 lstm_18/while/lstm_cell_18/add_1AddV2"lstm_18/while/lstm_cell_18/mul:z:0$lstm_18/while/lstm_cell_18/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2"
 lstm_18/while/lstm_cell_18/add_1Д
$lstm_18/while/lstm_cell_18/Sigmoid_2Sigmoid)lstm_18/while/lstm_cell_18/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџK2&
$lstm_18/while/lstm_cell_18/Sigmoid_2І
!lstm_18/while/lstm_cell_18/Relu_1Relu$lstm_18/while/lstm_cell_18/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2#
!lstm_18/while/lstm_cell_18/Relu_1и
 lstm_18/while/lstm_cell_18/mul_2Mul(lstm_18/while/lstm_cell_18/Sigmoid_2:y:0/lstm_18/while/lstm_cell_18/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2"
 lstm_18/while/lstm_cell_18/mul_2
2lstm_18/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_18_while_placeholder_1lstm_18_while_placeholder$lstm_18/while/lstm_cell_18/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_18/while/TensorArrayV2Write/TensorListSetIteml
lstm_18/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_18/while/add/y
lstm_18/while/addAddV2lstm_18_while_placeholderlstm_18/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_18/while/addp
lstm_18/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_18/while/add_1/y
lstm_18/while/add_1AddV2(lstm_18_while_lstm_18_while_loop_counterlstm_18/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_18/while/add_1
lstm_18/while/IdentityIdentitylstm_18/while/add_1:z:02^lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOp1^lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp3^lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_18/while/Identity­
lstm_18/while/Identity_1Identity.lstm_18_while_lstm_18_while_maximum_iterations2^lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOp1^lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp3^lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_18/while/Identity_1
lstm_18/while/Identity_2Identitylstm_18/while/add:z:02^lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOp1^lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp3^lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_18/while/Identity_2С
lstm_18/while/Identity_3IdentityBlstm_18/while/TensorArrayV2Write/TensorListSetItem:output_handle:02^lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOp1^lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp3^lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_18/while/Identity_3Д
lstm_18/while/Identity_4Identity$lstm_18/while/lstm_cell_18/mul_2:z:02^lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOp1^lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp3^lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_18/while/Identity_4Д
lstm_18/while/Identity_5Identity$lstm_18/while/lstm_cell_18/add_1:z:02^lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOp1^lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp3^lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_18/while/Identity_5"9
lstm_18_while_identitylstm_18/while/Identity:output:0"=
lstm_18_while_identity_1!lstm_18/while/Identity_1:output:0"=
lstm_18_while_identity_2!lstm_18/while/Identity_2:output:0"=
lstm_18_while_identity_3!lstm_18/while/Identity_3:output:0"=
lstm_18_while_identity_4!lstm_18/while/Identity_4:output:0"=
lstm_18_while_identity_5!lstm_18/while/Identity_5:output:0"P
%lstm_18_while_lstm_18_strided_slice_1'lstm_18_while_lstm_18_strided_slice_1_0"z
:lstm_18_while_lstm_cell_18_biasadd_readvariableop_resource<lstm_18_while_lstm_cell_18_biasadd_readvariableop_resource_0"|
;lstm_18_while_lstm_cell_18_matmul_1_readvariableop_resource=lstm_18_while_lstm_cell_18_matmul_1_readvariableop_resource_0"x
9lstm_18_while_lstm_cell_18_matmul_readvariableop_resource;lstm_18_while_lstm_cell_18_matmul_readvariableop_resource_0"Ш
alstm_18_while_tensorarrayv2read_tensorlistgetitem_lstm_18_tensorarrayunstack_tensorlistfromtensorclstm_18_while_tensorarrayv2read_tensorlistgetitem_lstm_18_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџK:џџџџџџџџџK: : :::2f
1lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOp1lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOp2d
0lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp0lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp2h
2lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOp2lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџK:-)
'
_output_shapes
:џџџџџџџџџK:

_output_shapes
: :

_output_shapes
: 
Е
Э
while_cond_44729837
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_44729837___redundant_placeholder06
2while_while_cond_44729837___redundant_placeholder16
2while_while_cond_44729837___redundant_placeholder26
2while_while_cond_44729837___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџK:џџџџџџџџџK: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџK:-)
'
_output_shapes
:џџџџџџџџџK:

_output_shapes
: :

_output_shapes
:
Е
Э
while_cond_44725918
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_44725918___redundant_placeholder06
2while_while_cond_44725918___redundant_placeholder16
2while_while_cond_44725918___redundant_placeholder26
2while_while_cond_44725918___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџK:џџџџџџџџџK: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџK:-)
'
_output_shapes
:џџџџџџџџџK:

_output_shapes
: :

_output_shapes
:
+
ћ
E__inference_model_9_layer_call_and_return_conditional_losses_44728646

inputs
lstm_18_44728606
lstm_18_44728608
lstm_18_44728610
gru_9_44728613
gru_9_44728615
gru_9_44728617
lstm_19_44728622
lstm_19_44728624
lstm_19_44728626
dense_27_44728629
dense_27_44728631
dense_28_44728634
dense_28_44728636
dense_29_44728640
dense_29_44728642
identityЂ dense_27/StatefulPartitionedCallЂ dense_28/StatefulPartitionedCallЂ dense_29/StatefulPartitionedCallЂgru_9/StatefulPartitionedCallЂlstm_18/StatefulPartitionedCallЂlstm_19/StatefulPartitionedCallЙ
lstm_18/StatefulPartitionedCallStatefulPartitionedCallinputslstm_18_44728606lstm_18_44728608lstm_18_44728610*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_lstm_18_layer_call_and_return_conditional_losses_447276102!
lstm_18/StatefulPartitionedCall 
gru_9/StatefulPartitionedCallStatefulPartitionedCallinputsgru_9_44728613gru_9_44728615gru_9_44728617*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_gru_9_layer_call_and_return_conditional_losses_447279572
gru_9/StatefulPartitionedCall
dropout_18/PartitionedCallPartitionedCall(lstm_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dropout_18_layer_call_and_return_conditional_losses_447280042
dropout_18/PartitionedCallџ
dropout_19/PartitionedCallPartitionedCall&gru_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dropout_19_layer_call_and_return_conditional_losses_447280342
dropout_19/PartitionedCallЩ
lstm_19/StatefulPartitionedCallStatefulPartitionedCall#dropout_18/PartitionedCall:output:0lstm_19_44728622lstm_19_44728624lstm_19_44728626*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_lstm_19_layer_call_and_return_conditional_losses_447283522!
lstm_19/StatefulPartitionedCallП
 dense_27/StatefulPartitionedCallStatefulPartitionedCall(lstm_19/StatefulPartitionedCall:output:0dense_27_44728629dense_27_44728631*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_27_layer_call_and_return_conditional_losses_447283932"
 dense_27/StatefulPartitionedCallК
 dense_28/StatefulPartitionedCallStatefulPartitionedCall#dropout_19/PartitionedCall:output:0dense_28_44728634dense_28_44728636*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_28_layer_call_and_return_conditional_losses_447284202"
 dense_28/StatefulPartitionedCallЗ
concatenate_9/PartitionedCallPartitionedCall)dense_27/StatefulPartitionedCall:output:0)dense_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_concatenate_9_layer_call_and_return_conditional_losses_447284432
concatenate_9/PartitionedCallН
 dense_29/StatefulPartitionedCallStatefulPartitionedCall&concatenate_9/PartitionedCall:output:0dense_29_44728640dense_29_44728642*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_29_layer_call_and_return_conditional_losses_447284622"
 dense_29/StatefulPartitionedCallЪ
IdentityIdentity)dense_29/StatefulPartitionedCall:output:0!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall^gru_9/StatefulPartitionedCall ^lstm_18/StatefulPartitionedCall ^lstm_19/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:џџџџџџџџџџџџџџџџџџ:::::::::::::::2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2>
gru_9/StatefulPartitionedCallgru_9/StatefulPartitionedCall2B
lstm_18/StatefulPartitionedCalllstm_18/StatefulPartitionedCall2B
lstm_19/StatefulPartitionedCalllstm_19/StatefulPartitionedCall:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
C

while_body_44727525
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_18_matmul_readvariableop_resource_09
5while_lstm_cell_18_matmul_1_readvariableop_resource_08
4while_lstm_cell_18_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_18_matmul_readvariableop_resource7
3while_lstm_cell_18_matmul_1_readvariableop_resource6
2while_lstm_cell_18_biasadd_readvariableop_resourceЂ)while/lstm_cell_18/BiasAdd/ReadVariableOpЂ(while/lstm_cell_18/MatMul/ReadVariableOpЂ*while/lstm_cell_18/MatMul_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЩ
(while/lstm_cell_18/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_18_matmul_readvariableop_resource_0*
_output_shapes
:	Ќ*
dtype02*
(while/lstm_cell_18/MatMul/ReadVariableOpз
while/lstm_cell_18/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_18/MatMulЯ
*while/lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_18_matmul_1_readvariableop_resource_0*
_output_shapes
:	KЌ*
dtype02,
*while/lstm_cell_18/MatMul_1/ReadVariableOpР
while/lstm_cell_18/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_18/MatMul_1И
while/lstm_cell_18/addAddV2#while/lstm_cell_18/MatMul:product:0%while/lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_18/addШ
)while/lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_18_biasadd_readvariableop_resource_0*
_output_shapes	
:Ќ*
dtype02+
)while/lstm_cell_18/BiasAdd/ReadVariableOpХ
while/lstm_cell_18/BiasAddBiasAddwhile/lstm_cell_18/add:z:01while/lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_18/BiasAddv
while/lstm_cell_18/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_18/Const
"while/lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_18/split/split_dim
while/lstm_cell_18/splitSplit+while/lstm_cell_18/split/split_dim:output:0#while/lstm_cell_18/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*
	num_split2
while/lstm_cell_18/split
while/lstm_cell_18/SigmoidSigmoid!while/lstm_cell_18/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/Sigmoid
while/lstm_cell_18/Sigmoid_1Sigmoid!while/lstm_cell_18/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/Sigmoid_1 
while/lstm_cell_18/mulMul while/lstm_cell_18/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/mul
while/lstm_cell_18/ReluRelu!while/lstm_cell_18/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/ReluД
while/lstm_cell_18/mul_1Mulwhile/lstm_cell_18/Sigmoid:y:0%while/lstm_cell_18/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/mul_1Љ
while/lstm_cell_18/add_1AddV2while/lstm_cell_18/mul:z:0while/lstm_cell_18/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/add_1
while/lstm_cell_18/Sigmoid_2Sigmoid!while/lstm_cell_18/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/Sigmoid_2
while/lstm_cell_18/Relu_1Reluwhile/lstm_cell_18/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/Relu_1И
while/lstm_cell_18/mul_2Mul while/lstm_cell_18/Sigmoid_2:y:0'while/lstm_cell_18/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_18/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1т
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_18/BiasAdd/ReadVariableOp)^while/lstm_cell_18/MatMul/ReadVariableOp+^while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityѕ
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_18/BiasAdd/ReadVariableOp)^while/lstm_cell_18/MatMul/ReadVariableOp+^while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1ф
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_18/BiasAdd/ReadVariableOp)^while/lstm_cell_18/MatMul/ReadVariableOp+^while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_18/BiasAdd/ReadVariableOp)^while/lstm_cell_18/MatMul/ReadVariableOp+^while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_18/mul_2:z:0*^while/lstm_cell_18/BiasAdd/ReadVariableOp)^while/lstm_cell_18/MatMul/ReadVariableOp+^while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_18/add_1:z:0*^while/lstm_cell_18/BiasAdd/ReadVariableOp)^while/lstm_cell_18/MatMul/ReadVariableOp+^while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_18_biasadd_readvariableop_resource4while_lstm_cell_18_biasadd_readvariableop_resource_0"l
3while_lstm_cell_18_matmul_1_readvariableop_resource5while_lstm_cell_18_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_18_matmul_readvariableop_resource3while_lstm_cell_18_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџK:џџџџџџџџџK: : :::2V
)while/lstm_cell_18/BiasAdd/ReadVariableOp)while/lstm_cell_18/BiasAdd/ReadVariableOp2T
(while/lstm_cell_18/MatMul/ReadVariableOp(while/lstm_cell_18/MatMul/ReadVariableOp2X
*while/lstm_cell_18/MatMul_1/ReadVariableOp*while/lstm_cell_18/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџK:-)
'
_output_shapes
:џџџџџџџџџK:

_output_shapes
: :

_output_shapes
: 


(__inference_gru_9_layer_call_fn_44730782

inputs
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_gru_9_layer_call_and_return_conditional_losses_447277982
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
.
Ч
E__inference_model_9_layer_call_and_return_conditional_losses_44728479
input_10
lstm_18_44727633
lstm_18_44727635
lstm_18_44727637
gru_9_44727980
gru_9_44727982
gru_9_44727984
lstm_19_44728375
lstm_19_44728377
lstm_19_44728379
dense_27_44728404
dense_27_44728406
dense_28_44728431
dense_28_44728433
dense_29_44728473
dense_29_44728475
identityЂ dense_27/StatefulPartitionedCallЂ dense_28/StatefulPartitionedCallЂ dense_29/StatefulPartitionedCallЂ"dropout_18/StatefulPartitionedCallЂ"dropout_19/StatefulPartitionedCallЂgru_9/StatefulPartitionedCallЂlstm_18/StatefulPartitionedCallЂlstm_19/StatefulPartitionedCallЛ
lstm_18/StatefulPartitionedCallStatefulPartitionedCallinput_10lstm_18_44727633lstm_18_44727635lstm_18_44727637*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_lstm_18_layer_call_and_return_conditional_losses_447274572!
lstm_18/StatefulPartitionedCallЂ
gru_9/StatefulPartitionedCallStatefulPartitionedCallinput_10gru_9_44727980gru_9_44727982gru_9_44727984*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_gru_9_layer_call_and_return_conditional_losses_447277982
gru_9/StatefulPartitionedCallІ
"dropout_18/StatefulPartitionedCallStatefulPartitionedCall(lstm_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dropout_18_layer_call_and_return_conditional_losses_447279992$
"dropout_18/StatefulPartitionedCallМ
"dropout_19/StatefulPartitionedCallStatefulPartitionedCall&gru_9/StatefulPartitionedCall:output:0#^dropout_18/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dropout_19_layer_call_and_return_conditional_losses_447280292$
"dropout_19/StatefulPartitionedCallб
lstm_19/StatefulPartitionedCallStatefulPartitionedCall+dropout_18/StatefulPartitionedCall:output:0lstm_19_44728375lstm_19_44728377lstm_19_44728379*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_lstm_19_layer_call_and_return_conditional_losses_447281992!
lstm_19/StatefulPartitionedCallП
 dense_27/StatefulPartitionedCallStatefulPartitionedCall(lstm_19/StatefulPartitionedCall:output:0dense_27_44728404dense_27_44728406*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_27_layer_call_and_return_conditional_losses_447283932"
 dense_27/StatefulPartitionedCallТ
 dense_28/StatefulPartitionedCallStatefulPartitionedCall+dropout_19/StatefulPartitionedCall:output:0dense_28_44728431dense_28_44728433*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_28_layer_call_and_return_conditional_losses_447284202"
 dense_28/StatefulPartitionedCallЗ
concatenate_9/PartitionedCallPartitionedCall)dense_27/StatefulPartitionedCall:output:0)dense_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_concatenate_9_layer_call_and_return_conditional_losses_447284432
concatenate_9/PartitionedCallН
 dense_29/StatefulPartitionedCallStatefulPartitionedCall&concatenate_9/PartitionedCall:output:0dense_29_44728473dense_29_44728475*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_29_layer_call_and_return_conditional_losses_447284622"
 dense_29/StatefulPartitionedCall
IdentityIdentity)dense_29/StatefulPartitionedCall:output:0!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall#^dropout_18/StatefulPartitionedCall#^dropout_19/StatefulPartitionedCall^gru_9/StatefulPartitionedCall ^lstm_18/StatefulPartitionedCall ^lstm_19/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:џџџџџџџџџџџџџџџџџџ:::::::::::::::2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2H
"dropout_18/StatefulPartitionedCall"dropout_18/StatefulPartitionedCall2H
"dropout_19/StatefulPartitionedCall"dropout_19/StatefulPartitionedCall2>
gru_9/StatefulPartitionedCallgru_9/StatefulPartitionedCall2B
lstm_18/StatefulPartitionedCalllstm_18/StatefulPartitionedCall2B
lstm_19/StatefulPartitionedCalllstm_19/StatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
input_10
C

while_body_44728267
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_19_matmul_readvariableop_resource_09
5while_lstm_cell_19_matmul_1_readvariableop_resource_08
4while_lstm_cell_19_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_19_matmul_readvariableop_resource7
3while_lstm_cell_19_matmul_1_readvariableop_resource6
2while_lstm_cell_19_biasadd_readvariableop_resourceЂ)while/lstm_cell_19/BiasAdd/ReadVariableOpЂ(while/lstm_cell_19/MatMul/ReadVariableOpЂ*while/lstm_cell_19/MatMul_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџK*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЩ
(while/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_19_matmul_readvariableop_resource_0*
_output_shapes
:	KШ*
dtype02*
(while/lstm_cell_19/MatMul/ReadVariableOpз
while/lstm_cell_19/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_19/MatMulЯ
*while/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_19_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02,
*while/lstm_cell_19/MatMul_1/ReadVariableOpР
while/lstm_cell_19/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_19/MatMul_1И
while/lstm_cell_19/addAddV2#while/lstm_cell_19/MatMul:product:0%while/lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_19/addШ
)while/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_19_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02+
)while/lstm_cell_19/BiasAdd/ReadVariableOpХ
while/lstm_cell_19/BiasAddBiasAddwhile/lstm_cell_19/add:z:01while/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_19/BiasAddv
while/lstm_cell_19/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_19/Const
"while/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_19/split/split_dim
while/lstm_cell_19/splitSplit+while/lstm_cell_19/split/split_dim:output:0#while/lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_19/split
while/lstm_cell_19/SigmoidSigmoid!while/lstm_cell_19/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/Sigmoid
while/lstm_cell_19/Sigmoid_1Sigmoid!while/lstm_cell_19/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/Sigmoid_1 
while/lstm_cell_19/mulMul while/lstm_cell_19/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/mul
while/lstm_cell_19/ReluRelu!while/lstm_cell_19/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/ReluД
while/lstm_cell_19/mul_1Mulwhile/lstm_cell_19/Sigmoid:y:0%while/lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/mul_1Љ
while/lstm_cell_19/add_1AddV2while/lstm_cell_19/mul:z:0while/lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/add_1
while/lstm_cell_19/Sigmoid_2Sigmoid!while/lstm_cell_19/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/Sigmoid_2
while/lstm_cell_19/Relu_1Reluwhile/lstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/Relu_1И
while/lstm_cell_19/mul_2Mul while/lstm_cell_19/Sigmoid_2:y:0'while/lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_19/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1т
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityѕ
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1ф
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_19/mul_2:z:0*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_19/add_1:z:0*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_19_biasadd_readvariableop_resource4while_lstm_cell_19_biasadd_readvariableop_resource_0"l
3while_lstm_cell_19_matmul_1_readvariableop_resource5while_lstm_cell_19_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_19_matmul_readvariableop_resource3while_lstm_cell_19_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : :::2V
)while/lstm_cell_19/BiasAdd/ReadVariableOp)while/lstm_cell_19/BiasAdd/ReadVariableOp2T
(while/lstm_cell_19/MatMul/ReadVariableOp(while/lstm_cell_19/MatMul/ReadVariableOp2X
*while/lstm_cell_19/MatMul_1/ReadVariableOp*while/lstm_cell_19/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
Ы
­
H__inference_gru_cell_9_layer_call_and_return_conditional_losses_44726241

inputs

states
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1ЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split/split_dimЌ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   џџџџ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split_1/split_dimг
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
Relu\
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:џџџџџџџџџ22
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
subd
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
add_3 
IdentityIdentity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22

IdentityЄ

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ2:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_namestates
Э[
і
E__inference_lstm_19_layer_call_and_return_conditional_losses_44731439
inputs_0/
+lstm_cell_19_matmul_readvariableop_resource1
-lstm_cell_19_matmul_1_readvariableop_resource0
,lstm_cell_19_biasadd_readvariableop_resource
identityЂ#lstm_cell_19/BiasAdd/ReadVariableOpЂ"lstm_cell_19/MatMul/ReadVariableOpЂ$lstm_cell_19/MatMul_1/ReadVariableOpЂwhileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџK*
shrink_axis_mask2
strided_slice_2Е
"lstm_cell_19/MatMul/ReadVariableOpReadVariableOp+lstm_cell_19_matmul_readvariableop_resource*
_output_shapes
:	KШ*
dtype02$
"lstm_cell_19/MatMul/ReadVariableOp­
lstm_cell_19/MatMulMatMulstrided_slice_2:output:0*lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_19/MatMulЛ
$lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_19_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02&
$lstm_cell_19/MatMul_1/ReadVariableOpЉ
lstm_cell_19/MatMul_1MatMulzeros:output:0,lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_19/MatMul_1 
lstm_cell_19/addAddV2lstm_cell_19/MatMul:product:0lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_19/addД
#lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_19_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02%
#lstm_cell_19/BiasAdd/ReadVariableOp­
lstm_cell_19/BiasAddBiasAddlstm_cell_19/add:z:0+lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_19/BiasAddj
lstm_cell_19/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_19/Const~
lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_19/split/split_dimѓ
lstm_cell_19/splitSplit%lstm_cell_19/split/split_dim:output:0lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_19/split
lstm_cell_19/SigmoidSigmoidlstm_cell_19/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/Sigmoid
lstm_cell_19/Sigmoid_1Sigmoidlstm_cell_19/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/Sigmoid_1
lstm_cell_19/mulMullstm_cell_19/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/mul}
lstm_cell_19/ReluRelulstm_cell_19/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/Relu
lstm_cell_19/mul_1Mullstm_cell_19/Sigmoid:y:0lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/mul_1
lstm_cell_19/add_1AddV2lstm_cell_19/mul:z:0lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/add_1
lstm_cell_19/Sigmoid_2Sigmoidlstm_cell_19/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/Sigmoid_2|
lstm_cell_19/Relu_1Relulstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/Relu_1 
lstm_cell_19/mul_2Mullstm_cell_19/Sigmoid_2:y:0!lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterђ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_19_matmul_readvariableop_resource-lstm_cell_19_matmul_1_readvariableop_resource,lstm_cell_19_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_44731354*
condR
while_cond_44731353*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeц
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_19/BiasAdd/ReadVariableOp#^lstm_cell_19/MatMul/ReadVariableOp%^lstm_cell_19/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџK:::2J
#lstm_cell_19/BiasAdd/ReadVariableOp#lstm_cell_19/BiasAdd/ReadVariableOp2H
"lstm_cell_19/MatMul/ReadVariableOp"lstm_cell_19/MatMul/ReadVariableOp2L
$lstm_cell_19/MatMul_1/ReadVariableOp$lstm_cell_19/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK
"
_user_specified_name
inputs/0
ђ<
к
C__inference_gru_9_layer_call_and_return_conditional_losses_44726682

inputs
gru_cell_9_44726606
gru_cell_9_44726608
gru_cell_9_44726610
identityЂ"gru_cell_9/StatefulPartitionedCallЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2ѕ
"gru_cell_9/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_9_44726606gru_cell_9_44726608gru_cell_9_44726610*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ2:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_gru_cell_9_layer_call_and_return_conditional_losses_447262412$
"gru_cell_9/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterю
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_9_44726606gru_cell_9_44726608gru_cell_9_44726610*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_44726618*
condR
while_cond_44726617*8
output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime
IdentityIdentitystrided_slice_3:output:0#^gru_cell_9/StatefulPartitionedCall^while*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2H
"gru_cell_9/StatefulPartitionedCall"gru_cell_9/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
№	
п
F__inference_dense_28_layer_call_and_return_conditional_losses_44731847

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
І

э
lstm_19_while_cond_44729111,
(lstm_19_while_lstm_19_while_loop_counter2
.lstm_19_while_lstm_19_while_maximum_iterations
lstm_19_while_placeholder
lstm_19_while_placeholder_1
lstm_19_while_placeholder_2
lstm_19_while_placeholder_3.
*lstm_19_while_less_lstm_19_strided_slice_1F
Blstm_19_while_lstm_19_while_cond_44729111___redundant_placeholder0F
Blstm_19_while_lstm_19_while_cond_44729111___redundant_placeholder1F
Blstm_19_while_lstm_19_while_cond_44729111___redundant_placeholder2F
Blstm_19_while_lstm_19_while_cond_44729111___redundant_placeholder3
lstm_19_while_identity

lstm_19/while/LessLesslstm_19_while_placeholder*lstm_19_while_less_lstm_19_strided_slice_1*
T0*
_output_shapes
: 2
lstm_19/while/Lessu
lstm_19/while/IdentityIdentitylstm_19/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_19/while/Identity"9
lstm_19_while_identitylstm_19/while/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџ2:џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
t
в
!__inference__traced_save_44732387
file_prefix.
*savev2_dense_27_kernel_read_readvariableop,
(savev2_dense_27_bias_read_readvariableop.
*savev2_dense_28_kernel_read_readvariableop,
(savev2_dense_28_bias_read_readvariableop.
*savev2_dense_29_kernel_read_readvariableop,
(savev2_dense_29_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_lstm_18_lstm_cell_18_kernel_read_readvariableopD
@savev2_lstm_18_lstm_cell_18_recurrent_kernel_read_readvariableop8
4savev2_lstm_18_lstm_cell_18_bias_read_readvariableop6
2savev2_gru_9_gru_cell_9_kernel_read_readvariableop@
<savev2_gru_9_gru_cell_9_recurrent_kernel_read_readvariableop4
0savev2_gru_9_gru_cell_9_bias_read_readvariableop:
6savev2_lstm_19_lstm_cell_19_kernel_read_readvariableopD
@savev2_lstm_19_lstm_cell_19_recurrent_kernel_read_readvariableop8
4savev2_lstm_19_lstm_cell_19_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop5
1savev2_adam_dense_27_kernel_m_read_readvariableop3
/savev2_adam_dense_27_bias_m_read_readvariableop5
1savev2_adam_dense_28_kernel_m_read_readvariableop3
/savev2_adam_dense_28_bias_m_read_readvariableop5
1savev2_adam_dense_29_kernel_m_read_readvariableop3
/savev2_adam_dense_29_bias_m_read_readvariableopA
=savev2_adam_lstm_18_lstm_cell_18_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_18_lstm_cell_18_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_18_lstm_cell_18_bias_m_read_readvariableop=
9savev2_adam_gru_9_gru_cell_9_kernel_m_read_readvariableopG
Csavev2_adam_gru_9_gru_cell_9_recurrent_kernel_m_read_readvariableop;
7savev2_adam_gru_9_gru_cell_9_bias_m_read_readvariableopA
=savev2_adam_lstm_19_lstm_cell_19_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_19_lstm_cell_19_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_19_lstm_cell_19_bias_m_read_readvariableop5
1savev2_adam_dense_27_kernel_v_read_readvariableop3
/savev2_adam_dense_27_bias_v_read_readvariableop5
1savev2_adam_dense_28_kernel_v_read_readvariableop3
/savev2_adam_dense_28_bias_v_read_readvariableop5
1savev2_adam_dense_29_kernel_v_read_readvariableop3
/savev2_adam_dense_29_bias_v_read_readvariableopA
=savev2_adam_lstm_18_lstm_cell_18_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_18_lstm_cell_18_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_18_lstm_cell_18_bias_v_read_readvariableop=
9savev2_adam_gru_9_gru_cell_9_kernel_v_read_readvariableopG
Csavev2_adam_gru_9_gru_cell_9_recurrent_kernel_v_read_readvariableop;
7savev2_adam_gru_9_gru_cell_9_bias_v_read_readvariableopA
=savev2_adam_lstm_19_lstm_cell_19_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_19_lstm_cell_19_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_19_lstm_cell_19_bias_v_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameЖ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:9*
dtype0*Ш
valueОBЛ9B6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesћ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:9*
dtype0*
value|Bz9B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesі
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_27_kernel_read_readvariableop(savev2_dense_27_bias_read_readvariableop*savev2_dense_28_kernel_read_readvariableop(savev2_dense_28_bias_read_readvariableop*savev2_dense_29_kernel_read_readvariableop(savev2_dense_29_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_lstm_18_lstm_cell_18_kernel_read_readvariableop@savev2_lstm_18_lstm_cell_18_recurrent_kernel_read_readvariableop4savev2_lstm_18_lstm_cell_18_bias_read_readvariableop2savev2_gru_9_gru_cell_9_kernel_read_readvariableop<savev2_gru_9_gru_cell_9_recurrent_kernel_read_readvariableop0savev2_gru_9_gru_cell_9_bias_read_readvariableop6savev2_lstm_19_lstm_cell_19_kernel_read_readvariableop@savev2_lstm_19_lstm_cell_19_recurrent_kernel_read_readvariableop4savev2_lstm_19_lstm_cell_19_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop1savev2_adam_dense_27_kernel_m_read_readvariableop/savev2_adam_dense_27_bias_m_read_readvariableop1savev2_adam_dense_28_kernel_m_read_readvariableop/savev2_adam_dense_28_bias_m_read_readvariableop1savev2_adam_dense_29_kernel_m_read_readvariableop/savev2_adam_dense_29_bias_m_read_readvariableop=savev2_adam_lstm_18_lstm_cell_18_kernel_m_read_readvariableopGsavev2_adam_lstm_18_lstm_cell_18_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_18_lstm_cell_18_bias_m_read_readvariableop9savev2_adam_gru_9_gru_cell_9_kernel_m_read_readvariableopCsavev2_adam_gru_9_gru_cell_9_recurrent_kernel_m_read_readvariableop7savev2_adam_gru_9_gru_cell_9_bias_m_read_readvariableop=savev2_adam_lstm_19_lstm_cell_19_kernel_m_read_readvariableopGsavev2_adam_lstm_19_lstm_cell_19_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_19_lstm_cell_19_bias_m_read_readvariableop1savev2_adam_dense_27_kernel_v_read_readvariableop/savev2_adam_dense_27_bias_v_read_readvariableop1savev2_adam_dense_28_kernel_v_read_readvariableop/savev2_adam_dense_28_bias_v_read_readvariableop1savev2_adam_dense_29_kernel_v_read_readvariableop/savev2_adam_dense_29_bias_v_read_readvariableop=savev2_adam_lstm_18_lstm_cell_18_kernel_v_read_readvariableopGsavev2_adam_lstm_18_lstm_cell_18_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_18_lstm_cell_18_bias_v_read_readvariableop9savev2_adam_gru_9_gru_cell_9_kernel_v_read_readvariableopCsavev2_adam_gru_9_gru_cell_9_recurrent_kernel_v_read_readvariableop7savev2_adam_gru_9_gru_cell_9_bias_v_read_readvariableop=savev2_adam_lstm_19_lstm_cell_19_kernel_v_read_readvariableopGsavev2_adam_lstm_19_lstm_cell_19_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_19_lstm_cell_19_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *G
dtypes=
;29	2
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*а
_input_shapesО
Л: :2 : :2@:@:`:: : : : : :	Ќ:	KЌ:Ќ:	:	2:	:	KШ:	2Ш:Ш: : : : : : :2 : :2@:@:`::	Ќ:	KЌ:Ќ:	:	2:	:	KШ:	2Ш:Ш:2 : :2@:@:`::	Ќ:	KЌ:Ќ:	:	2:	:	KШ:	2Ш:Ш: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:2 : 

_output_shapes
: :$ 

_output_shapes

:2@: 

_output_shapes
:@:$ 

_output_shapes

:`: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	Ќ:%!

_output_shapes
:	KЌ:!

_output_shapes	
:Ќ:%!

_output_shapes
:	:%!

_output_shapes
:	2:%!

_output_shapes
:	:%!

_output_shapes
:	KШ:%!

_output_shapes
:	2Ш:!

_output_shapes	
:Ш:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:2 : 

_output_shapes
: :$ 

_output_shapes

:2@: 

_output_shapes
:@:$ 

_output_shapes

:`:  

_output_shapes
::%!!

_output_shapes
:	Ќ:%"!

_output_shapes
:	KЌ:!#

_output_shapes	
:Ќ:%$!

_output_shapes
:	:%%!

_output_shapes
:	2:%&!

_output_shapes
:	:%'!

_output_shapes
:	KШ:%(!

_output_shapes
:	2Ш:!)

_output_shapes	
:Ш:$* 

_output_shapes

:2 : +

_output_shapes
: :$, 

_output_shapes

:2@: -

_output_shapes
:@:$. 

_output_shapes

:`: /

_output_shapes
::%0!

_output_shapes
:	Ќ:%1!

_output_shapes
:	KЌ:!2

_output_shapes	
:Ќ:%3!

_output_shapes
:	:%4!

_output_shapes
:	2:%5!

_output_shapes
:	:%6!

_output_shapes
:	KШ:%7!

_output_shapes
:	2Ш:!8

_output_shapes	
:Ш:9

_output_shapes
: 
Ђ

*__inference_lstm_19_layer_call_fn_44731450
inputs_0
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_lstm_19_layer_call_and_return_conditional_losses_447271602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџK:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK
"
_user_specified_name
inputs/0
C

while_body_44731354
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_19_matmul_readvariableop_resource_09
5while_lstm_cell_19_matmul_1_readvariableop_resource_08
4while_lstm_cell_19_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_19_matmul_readvariableop_resource7
3while_lstm_cell_19_matmul_1_readvariableop_resource6
2while_lstm_cell_19_biasadd_readvariableop_resourceЂ)while/lstm_cell_19/BiasAdd/ReadVariableOpЂ(while/lstm_cell_19/MatMul/ReadVariableOpЂ*while/lstm_cell_19/MatMul_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџK*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЩ
(while/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_19_matmul_readvariableop_resource_0*
_output_shapes
:	KШ*
dtype02*
(while/lstm_cell_19/MatMul/ReadVariableOpз
while/lstm_cell_19/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_19/MatMulЯ
*while/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_19_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02,
*while/lstm_cell_19/MatMul_1/ReadVariableOpР
while/lstm_cell_19/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_19/MatMul_1И
while/lstm_cell_19/addAddV2#while/lstm_cell_19/MatMul:product:0%while/lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_19/addШ
)while/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_19_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02+
)while/lstm_cell_19/BiasAdd/ReadVariableOpХ
while/lstm_cell_19/BiasAddBiasAddwhile/lstm_cell_19/add:z:01while/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_19/BiasAddv
while/lstm_cell_19/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_19/Const
"while/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_19/split/split_dim
while/lstm_cell_19/splitSplit+while/lstm_cell_19/split/split_dim:output:0#while/lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_19/split
while/lstm_cell_19/SigmoidSigmoid!while/lstm_cell_19/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/Sigmoid
while/lstm_cell_19/Sigmoid_1Sigmoid!while/lstm_cell_19/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/Sigmoid_1 
while/lstm_cell_19/mulMul while/lstm_cell_19/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/mul
while/lstm_cell_19/ReluRelu!while/lstm_cell_19/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/ReluД
while/lstm_cell_19/mul_1Mulwhile/lstm_cell_19/Sigmoid:y:0%while/lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/mul_1Љ
while/lstm_cell_19/add_1AddV2while/lstm_cell_19/mul:z:0while/lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/add_1
while/lstm_cell_19/Sigmoid_2Sigmoid!while/lstm_cell_19/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/Sigmoid_2
while/lstm_cell_19/Relu_1Reluwhile/lstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/Relu_1И
while/lstm_cell_19/mul_2Mul while/lstm_cell_19/Sigmoid_2:y:0'while/lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_19/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1т
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityѕ
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1ф
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_19/mul_2:z:0*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_19/add_1:z:0*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_19_biasadd_readvariableop_resource4while_lstm_cell_19_biasadd_readvariableop_resource_0"l
3while_lstm_cell_19_matmul_1_readvariableop_resource5while_lstm_cell_19_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_19_matmul_readvariableop_resource3while_lstm_cell_19_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : :::2V
)while/lstm_cell_19/BiasAdd/ReadVariableOp)while/lstm_cell_19/BiasAdd/ReadVariableOp2T
(while/lstm_cell_19/MatMul/ReadVariableOp(while/lstm_cell_19/MatMul/ReadVariableOp2X
*while/lstm_cell_19/MatMul_1/ReadVariableOp*while/lstm_cell_19/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
Х[
є
E__inference_lstm_19_layer_call_and_return_conditional_losses_44728199

inputs/
+lstm_cell_19_matmul_readvariableop_resource1
-lstm_cell_19_matmul_1_readvariableop_resource0
,lstm_cell_19_biasadd_readvariableop_resource
identityЂ#lstm_cell_19/BiasAdd/ReadVariableOpЂ"lstm_cell_19/MatMul/ReadVariableOpЂ$lstm_cell_19/MatMul_1/ReadVariableOpЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџK*
shrink_axis_mask2
strided_slice_2Е
"lstm_cell_19/MatMul/ReadVariableOpReadVariableOp+lstm_cell_19_matmul_readvariableop_resource*
_output_shapes
:	KШ*
dtype02$
"lstm_cell_19/MatMul/ReadVariableOp­
lstm_cell_19/MatMulMatMulstrided_slice_2:output:0*lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_19/MatMulЛ
$lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_19_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02&
$lstm_cell_19/MatMul_1/ReadVariableOpЉ
lstm_cell_19/MatMul_1MatMulzeros:output:0,lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_19/MatMul_1 
lstm_cell_19/addAddV2lstm_cell_19/MatMul:product:0lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_19/addД
#lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_19_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02%
#lstm_cell_19/BiasAdd/ReadVariableOp­
lstm_cell_19/BiasAddBiasAddlstm_cell_19/add:z:0+lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_19/BiasAddj
lstm_cell_19/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_19/Const~
lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_19/split/split_dimѓ
lstm_cell_19/splitSplit%lstm_cell_19/split/split_dim:output:0lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_19/split
lstm_cell_19/SigmoidSigmoidlstm_cell_19/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/Sigmoid
lstm_cell_19/Sigmoid_1Sigmoidlstm_cell_19/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/Sigmoid_1
lstm_cell_19/mulMullstm_cell_19/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/mul}
lstm_cell_19/ReluRelulstm_cell_19/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/Relu
lstm_cell_19/mul_1Mullstm_cell_19/Sigmoid:y:0lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/mul_1
lstm_cell_19/add_1AddV2lstm_cell_19/mul:z:0lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/add_1
lstm_cell_19/Sigmoid_2Sigmoidlstm_cell_19/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/Sigmoid_2|
lstm_cell_19/Relu_1Relulstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/Relu_1 
lstm_cell_19/mul_2Mullstm_cell_19/Sigmoid_2:y:0!lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterђ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_19_matmul_readvariableop_resource-lstm_cell_19_matmul_1_readvariableop_resource,lstm_cell_19_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_44728114*
condR
while_cond_44728113*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeц
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_19/BiasAdd/ReadVariableOp#^lstm_cell_19/MatMul/ReadVariableOp%^lstm_cell_19/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџK:::2J
#lstm_cell_19/BiasAdd/ReadVariableOp#lstm_cell_19/BiasAdd/ReadVariableOp2H
"lstm_cell_19/MatMul/ReadVariableOp"lstm_cell_19/MatMul/ReadVariableOp2L
$lstm_cell_19/MatMul_1/ReadVariableOp$lstm_cell_19/MatMul_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK
 
_user_specified_nameinputs
C

while_body_44729838
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_18_matmul_readvariableop_resource_09
5while_lstm_cell_18_matmul_1_readvariableop_resource_08
4while_lstm_cell_18_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_18_matmul_readvariableop_resource7
3while_lstm_cell_18_matmul_1_readvariableop_resource6
2while_lstm_cell_18_biasadd_readvariableop_resourceЂ)while/lstm_cell_18/BiasAdd/ReadVariableOpЂ(while/lstm_cell_18/MatMul/ReadVariableOpЂ*while/lstm_cell_18/MatMul_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЩ
(while/lstm_cell_18/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_18_matmul_readvariableop_resource_0*
_output_shapes
:	Ќ*
dtype02*
(while/lstm_cell_18/MatMul/ReadVariableOpз
while/lstm_cell_18/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_18/MatMulЯ
*while/lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_18_matmul_1_readvariableop_resource_0*
_output_shapes
:	KЌ*
dtype02,
*while/lstm_cell_18/MatMul_1/ReadVariableOpР
while/lstm_cell_18/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_18/MatMul_1И
while/lstm_cell_18/addAddV2#while/lstm_cell_18/MatMul:product:0%while/lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_18/addШ
)while/lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_18_biasadd_readvariableop_resource_0*
_output_shapes	
:Ќ*
dtype02+
)while/lstm_cell_18/BiasAdd/ReadVariableOpХ
while/lstm_cell_18/BiasAddBiasAddwhile/lstm_cell_18/add:z:01while/lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_18/BiasAddv
while/lstm_cell_18/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_18/Const
"while/lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_18/split/split_dim
while/lstm_cell_18/splitSplit+while/lstm_cell_18/split/split_dim:output:0#while/lstm_cell_18/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*
	num_split2
while/lstm_cell_18/split
while/lstm_cell_18/SigmoidSigmoid!while/lstm_cell_18/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/Sigmoid
while/lstm_cell_18/Sigmoid_1Sigmoid!while/lstm_cell_18/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/Sigmoid_1 
while/lstm_cell_18/mulMul while/lstm_cell_18/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/mul
while/lstm_cell_18/ReluRelu!while/lstm_cell_18/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/ReluД
while/lstm_cell_18/mul_1Mulwhile/lstm_cell_18/Sigmoid:y:0%while/lstm_cell_18/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/mul_1Љ
while/lstm_cell_18/add_1AddV2while/lstm_cell_18/mul:z:0while/lstm_cell_18/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/add_1
while/lstm_cell_18/Sigmoid_2Sigmoid!while/lstm_cell_18/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/Sigmoid_2
while/lstm_cell_18/Relu_1Reluwhile/lstm_cell_18/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/Relu_1И
while/lstm_cell_18/mul_2Mul while/lstm_cell_18/Sigmoid_2:y:0'while/lstm_cell_18/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_18/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1т
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_18/BiasAdd/ReadVariableOp)^while/lstm_cell_18/MatMul/ReadVariableOp+^while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityѕ
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_18/BiasAdd/ReadVariableOp)^while/lstm_cell_18/MatMul/ReadVariableOp+^while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1ф
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_18/BiasAdd/ReadVariableOp)^while/lstm_cell_18/MatMul/ReadVariableOp+^while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_18/BiasAdd/ReadVariableOp)^while/lstm_cell_18/MatMul/ReadVariableOp+^while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_18/mul_2:z:0*^while/lstm_cell_18/BiasAdd/ReadVariableOp)^while/lstm_cell_18/MatMul/ReadVariableOp+^while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_18/add_1:z:0*^while/lstm_cell_18/BiasAdd/ReadVariableOp)^while/lstm_cell_18/MatMul/ReadVariableOp+^while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_18_biasadd_readvariableop_resource4while_lstm_cell_18_biasadd_readvariableop_resource_0"l
3while_lstm_cell_18_matmul_1_readvariableop_resource5while_lstm_cell_18_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_18_matmul_readvariableop_resource3while_lstm_cell_18_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџK:џџџџџџџџџK: : :::2V
)while/lstm_cell_18/BiasAdd/ReadVariableOp)while/lstm_cell_18/BiasAdd/ReadVariableOp2T
(while/lstm_cell_18/MatMul/ReadVariableOp(while/lstm_cell_18/MatMul/ReadVariableOp2X
*while/lstm_cell_18/MatMul_1/ReadVariableOp*while/lstm_cell_18/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџK:-)
'
_output_shapes
:џџџџџџџџџK:

_output_shapes
: :

_output_shapes
: 
Щ[
є
E__inference_lstm_18_layer_call_and_return_conditional_losses_44729923

inputs/
+lstm_cell_18_matmul_readvariableop_resource1
-lstm_cell_18_matmul_1_readvariableop_resource0
,lstm_cell_18_biasadd_readvariableop_resource
identityЂ#lstm_cell_18/BiasAdd/ReadVariableOpЂ"lstm_cell_18/MatMul/ReadVariableOpЂ$lstm_cell_18/MatMul_1/ReadVariableOpЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2Е
"lstm_cell_18/MatMul/ReadVariableOpReadVariableOp+lstm_cell_18_matmul_readvariableop_resource*
_output_shapes
:	Ќ*
dtype02$
"lstm_cell_18/MatMul/ReadVariableOp­
lstm_cell_18/MatMulMatMulstrided_slice_2:output:0*lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_18/MatMulЛ
$lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_18_matmul_1_readvariableop_resource*
_output_shapes
:	KЌ*
dtype02&
$lstm_cell_18/MatMul_1/ReadVariableOpЉ
lstm_cell_18/MatMul_1MatMulzeros:output:0,lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_18/MatMul_1 
lstm_cell_18/addAddV2lstm_cell_18/MatMul:product:0lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_18/addД
#lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_18_biasadd_readvariableop_resource*
_output_shapes	
:Ќ*
dtype02%
#lstm_cell_18/BiasAdd/ReadVariableOp­
lstm_cell_18/BiasAddBiasAddlstm_cell_18/add:z:0+lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_18/BiasAddj
lstm_cell_18/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_18/Const~
lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_18/split/split_dimѓ
lstm_cell_18/splitSplit%lstm_cell_18/split/split_dim:output:0lstm_cell_18/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*
	num_split2
lstm_cell_18/split
lstm_cell_18/SigmoidSigmoidlstm_cell_18/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/Sigmoid
lstm_cell_18/Sigmoid_1Sigmoidlstm_cell_18/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/Sigmoid_1
lstm_cell_18/mulMullstm_cell_18/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/mul}
lstm_cell_18/ReluRelulstm_cell_18/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/Relu
lstm_cell_18/mul_1Mullstm_cell_18/Sigmoid:y:0lstm_cell_18/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/mul_1
lstm_cell_18/add_1AddV2lstm_cell_18/mul:z:0lstm_cell_18/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/add_1
lstm_cell_18/Sigmoid_2Sigmoidlstm_cell_18/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/Sigmoid_2|
lstm_cell_18/Relu_1Relulstm_cell_18/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/Relu_1 
lstm_cell_18/mul_2Mullstm_cell_18/Sigmoid_2:y:0!lstm_cell_18/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterђ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_18_matmul_readvariableop_resource-lstm_cell_18_matmul_1_readvariableop_resource,lstm_cell_18_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџK:џџџџџџџџџK: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_44729838*
condR
while_cond_44729837*K
output_shapes:
8: : : : :џџџџџџџџџK:џџџџџџџџџK: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџK*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeъ
IdentityIdentitytranspose_1:y:0$^lstm_cell_18/BiasAdd/ReadVariableOp#^lstm_cell_18/MatMul/ReadVariableOp%^lstm_cell_18/MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2J
#lstm_cell_18/BiasAdd/ReadVariableOp#lstm_cell_18/BiasAdd/ReadVariableOp2H
"lstm_cell_18/MatMul/ReadVariableOp"lstm_cell_18/MatMul/ReadVariableOp2L
$lstm_cell_18/MatMul_1/ReadVariableOp$lstm_cell_18/MatMul_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs


#model_9_lstm_19_while_cond_44725411<
8model_9_lstm_19_while_model_9_lstm_19_while_loop_counterB
>model_9_lstm_19_while_model_9_lstm_19_while_maximum_iterations%
!model_9_lstm_19_while_placeholder'
#model_9_lstm_19_while_placeholder_1'
#model_9_lstm_19_while_placeholder_2'
#model_9_lstm_19_while_placeholder_3>
:model_9_lstm_19_while_less_model_9_lstm_19_strided_slice_1V
Rmodel_9_lstm_19_while_model_9_lstm_19_while_cond_44725411___redundant_placeholder0V
Rmodel_9_lstm_19_while_model_9_lstm_19_while_cond_44725411___redundant_placeholder1V
Rmodel_9_lstm_19_while_model_9_lstm_19_while_cond_44725411___redundant_placeholder2V
Rmodel_9_lstm_19_while_model_9_lstm_19_while_cond_44725411___redundant_placeholder3"
model_9_lstm_19_while_identity
Р
model_9/lstm_19/while/LessLess!model_9_lstm_19_while_placeholder:model_9_lstm_19_while_less_model_9_lstm_19_strided_slice_1*
T0*
_output_shapes
: 2
model_9/lstm_19/while/Less
model_9/lstm_19/while/IdentityIdentitymodel_9/lstm_19/while/Less:z:0*
T0
*
_output_shapes
: 2 
model_9/lstm_19/while/Identity"I
model_9_lstm_19_while_identity'model_9/lstm_19/while/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџ2:џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
џF
Ї
while_body_44727708
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
*while_gru_cell_9_readvariableop_resource_05
1while_gru_cell_9_matmul_readvariableop_resource_07
3while_gru_cell_9_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
(while_gru_cell_9_readvariableop_resource3
/while_gru_cell_9_matmul_readvariableop_resource5
1while_gru_cell_9_matmul_1_readvariableop_resourceЂ&while/gru_cell_9/MatMul/ReadVariableOpЂ(while/gru_cell_9/MatMul_1/ReadVariableOpЂwhile/gru_cell_9/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЎ
while/gru_cell_9/ReadVariableOpReadVariableOp*while_gru_cell_9_readvariableop_resource_0*
_output_shapes
:	*
dtype02!
while/gru_cell_9/ReadVariableOp
while/gru_cell_9/unstackUnpack'while/gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
while/gru_cell_9/unstackУ
&while/gru_cell_9/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/gru_cell_9/MatMul/ReadVariableOpб
while/gru_cell_9/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
while/gru_cell_9/MatMulИ
while/gru_cell_9/BiasAddBiasAdd!while/gru_cell_9/MatMul:product:0!while/gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
while/gru_cell_9/BiasAddr
while/gru_cell_9/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_9/Const
 while/gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2"
 while/gru_cell_9/split/split_dim№
while/gru_cell_9/splitSplit)while/gru_cell_9/split/split_dim:output:0!while/gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/gru_cell_9/splitЩ
(while/gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes
:	2*
dtype02*
(while/gru_cell_9/MatMul_1/ReadVariableOpК
while/gru_cell_9/MatMul_1MatMulwhile_placeholder_20while/gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
while/gru_cell_9/MatMul_1О
while/gru_cell_9/BiasAdd_1BiasAdd#while/gru_cell_9/MatMul_1:product:0!while/gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2
while/gru_cell_9/BiasAdd_1
while/gru_cell_9/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   џџџџ2
while/gru_cell_9/Const_1
"while/gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2$
"while/gru_cell_9/split_1/split_dimЈ
while/gru_cell_9/split_1SplitV#while/gru_cell_9/BiasAdd_1:output:0!while/gru_cell_9/Const_1:output:0+while/gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/gru_cell_9/split_1Ћ
while/gru_cell_9/addAddV2while/gru_cell_9/split:output:0!while/gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/add
while/gru_cell_9/SigmoidSigmoidwhile/gru_cell_9/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/SigmoidЏ
while/gru_cell_9/add_1AddV2while/gru_cell_9/split:output:1!while/gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/add_1
while/gru_cell_9/Sigmoid_1Sigmoidwhile/gru_cell_9/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/Sigmoid_1Ј
while/gru_cell_9/mulMulwhile/gru_cell_9/Sigmoid_1:y:0!while/gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/mulІ
while/gru_cell_9/add_2AddV2while/gru_cell_9/split:output:2while/gru_cell_9/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/add_2
while/gru_cell_9/ReluReluwhile/gru_cell_9/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/Relu
while/gru_cell_9/mul_1Mulwhile/gru_cell_9/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/mul_1u
while/gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_9/sub/xЄ
while/gru_cell_9/subSubwhile/gru_cell_9/sub/x:output:0while/gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/subЈ
while/gru_cell_9/mul_2Mulwhile/gru_cell_9/sub:z:0#while/gru_cell_9/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/mul_2Ѓ
while/gru_cell_9/add_3AddV2while/gru_cell_9/mul_1:z:0while/gru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/gru_cell_9/add_3о
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_9/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1д
while/IdentityIdentitywhile/add_1:z:0'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityч
while/Identity_1Identitywhile_while_maximum_iterations'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1ж
while/Identity_2Identitywhile/add:z:0'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3є
while/Identity_4Identitywhile/gru_cell_9/add_3:z:0'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4"h
1while_gru_cell_9_matmul_1_readvariableop_resource3while_gru_cell_9_matmul_1_readvariableop_resource_0"d
/while_gru_cell_9_matmul_readvariableop_resource1while_gru_cell_9_matmul_readvariableop_resource_0"V
(while_gru_cell_9_readvariableop_resource*while_gru_cell_9_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ2: : :::2P
&while/gru_cell_9/MatMul/ReadVariableOp&while/gru_cell_9/MatMul/ReadVariableOp2T
(while/gru_cell_9/MatMul_1/ReadVariableOp(while/gru_cell_9/MatMul_1/ReadVariableOp2B
while/gru_cell_9/ReadVariableOpwhile/gru_cell_9/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
ї!
т
while_body_44726500
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_gru_cell_9_44726522_0
while_gru_cell_9_44726524_0
while_gru_cell_9_44726526_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_gru_cell_9_44726522
while_gru_cell_9_44726524
while_gru_cell_9_44726526Ђ(while/gru_cell_9/StatefulPartitionedCallУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЖ
(while/gru_cell_9/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_9_44726522_0while_gru_cell_9_44726524_0while_gru_cell_9_44726526_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ2:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_gru_cell_9_layer_call_and_return_conditional_losses_447262012*
(while/gru_cell_9/StatefulPartitionedCallѕ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/gru_cell_9/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0)^while/gru_cell_9/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/gru_cell_9/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/gru_cell_9/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2И
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/gru_cell_9/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Р
while/Identity_4Identity1while/gru_cell_9/StatefulPartitionedCall:output:1)^while/gru_cell_9/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4"8
while_gru_cell_9_44726522while_gru_cell_9_44726522_0"8
while_gru_cell_9_44726524while_gru_cell_9_44726524_0"8
while_gru_cell_9_44726526while_gru_cell_9_44726526_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ2: : :::2T
(while/gru_cell_9/StatefulPartitionedCall(while/gru_cell_9/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 


*__inference_lstm_19_layer_call_fn_44731778

inputs
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_lstm_19_layer_call_and_return_conditional_losses_447281992
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџK:::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK
 
_user_specified_nameinputs
Р%

while_body_44726051
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0!
while_lstm_cell_18_44726075_0!
while_lstm_cell_18_44726077_0!
while_lstm_cell_18_44726079_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_18_44726075
while_lstm_cell_18_44726077
while_lstm_cell_18_44726079Ђ*while/lstm_cell_18/StatefulPartitionedCallУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemь
*while/lstm_cell_18/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_18_44726075_0while_lstm_cell_18_44726077_0while_lstm_cell_18_44726079_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_lstm_cell_18_layer_call_and_return_conditional_losses_447256252,
*while/lstm_cell_18/StatefulPartitionedCallї
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_18/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0+^while/lstm_cell_18/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations+^while/lstm_cell_18/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0+^while/lstm_cell_18/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2К
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0+^while/lstm_cell_18/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Ф
while/Identity_4Identity3while/lstm_cell_18/StatefulPartitionedCall:output:1+^while/lstm_cell_18/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџK2
while/Identity_4Ф
while/Identity_5Identity3while/lstm_cell_18/StatefulPartitionedCall:output:2+^while/lstm_cell_18/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџK2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"<
while_lstm_cell_18_44726075while_lstm_cell_18_44726075_0"<
while_lstm_cell_18_44726077while_lstm_cell_18_44726077_0"<
while_lstm_cell_18_44726079while_lstm_cell_18_44726079_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџK:џџџџџџџџџK: : :::2X
*while/lstm_cell_18/StatefulPartitionedCall*while/lstm_cell_18/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџK:-)
'
_output_shapes
:џџџџџџџџџK:

_output_shapes
: :

_output_shapes
: 
	
п
F__inference_dense_29_layer_call_and_return_conditional_losses_44731879

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ`::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ`
 
_user_specified_nameinputs
C

while_body_44731682
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_19_matmul_readvariableop_resource_09
5while_lstm_cell_19_matmul_1_readvariableop_resource_08
4while_lstm_cell_19_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_19_matmul_readvariableop_resource7
3while_lstm_cell_19_matmul_1_readvariableop_resource6
2while_lstm_cell_19_biasadd_readvariableop_resourceЂ)while/lstm_cell_19/BiasAdd/ReadVariableOpЂ(while/lstm_cell_19/MatMul/ReadVariableOpЂ*while/lstm_cell_19/MatMul_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџK*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЩ
(while/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_19_matmul_readvariableop_resource_0*
_output_shapes
:	KШ*
dtype02*
(while/lstm_cell_19/MatMul/ReadVariableOpз
while/lstm_cell_19/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_19/MatMulЯ
*while/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_19_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02,
*while/lstm_cell_19/MatMul_1/ReadVariableOpР
while/lstm_cell_19/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_19/MatMul_1И
while/lstm_cell_19/addAddV2#while/lstm_cell_19/MatMul:product:0%while/lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_19/addШ
)while/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_19_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02+
)while/lstm_cell_19/BiasAdd/ReadVariableOpХ
while/lstm_cell_19/BiasAddBiasAddwhile/lstm_cell_19/add:z:01while/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
while/lstm_cell_19/BiasAddv
while/lstm_cell_19/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_19/Const
"while/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_19/split/split_dim
while/lstm_cell_19/splitSplit+while/lstm_cell_19/split/split_dim:output:0#while/lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
while/lstm_cell_19/split
while/lstm_cell_19/SigmoidSigmoid!while/lstm_cell_19/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/Sigmoid
while/lstm_cell_19/Sigmoid_1Sigmoid!while/lstm_cell_19/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/Sigmoid_1 
while/lstm_cell_19/mulMul while/lstm_cell_19/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/mul
while/lstm_cell_19/ReluRelu!while/lstm_cell_19/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/ReluД
while/lstm_cell_19/mul_1Mulwhile/lstm_cell_19/Sigmoid:y:0%while/lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/mul_1Љ
while/lstm_cell_19/add_1AddV2while/lstm_cell_19/mul:z:0while/lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/add_1
while/lstm_cell_19/Sigmoid_2Sigmoid!while/lstm_cell_19/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/Sigmoid_2
while/lstm_cell_19/Relu_1Reluwhile/lstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/Relu_1И
while/lstm_cell_19/mul_2Mul while/lstm_cell_19/Sigmoid_2:y:0'while/lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
while/lstm_cell_19/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_19/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1т
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityѕ
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1ф
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_19/mul_2:z:0*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_19/add_1:z:0*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_19_biasadd_readvariableop_resource4while_lstm_cell_19_biasadd_readvariableop_resource_0"l
3while_lstm_cell_19_matmul_1_readvariableop_resource5while_lstm_cell_19_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_19_matmul_readvariableop_resource3while_lstm_cell_19_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : :::2V
)while/lstm_cell_19/BiasAdd/ReadVariableOp)while/lstm_cell_19/BiasAdd/ReadVariableOp2T
(while/lstm_cell_19/MatMul/ReadVariableOp(while/lstm_cell_19/MatMul/ReadVariableOp2X
*while/lstm_cell_19/MatMul_1/ReadVariableOp*while/lstm_cell_19/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
б[
і
E__inference_lstm_18_layer_call_and_return_conditional_losses_44730251
inputs_0/
+lstm_cell_18_matmul_readvariableop_resource1
-lstm_cell_18_matmul_1_readvariableop_resource0
,lstm_cell_18_biasadd_readvariableop_resource
identityЂ#lstm_cell_18/BiasAdd/ReadVariableOpЂ"lstm_cell_18/MatMul/ReadVariableOpЂ$lstm_cell_18/MatMul_1/ReadVariableOpЂwhileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2Е
"lstm_cell_18/MatMul/ReadVariableOpReadVariableOp+lstm_cell_18_matmul_readvariableop_resource*
_output_shapes
:	Ќ*
dtype02$
"lstm_cell_18/MatMul/ReadVariableOp­
lstm_cell_18/MatMulMatMulstrided_slice_2:output:0*lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_18/MatMulЛ
$lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_18_matmul_1_readvariableop_resource*
_output_shapes
:	KЌ*
dtype02&
$lstm_cell_18/MatMul_1/ReadVariableOpЉ
lstm_cell_18/MatMul_1MatMulzeros:output:0,lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_18/MatMul_1 
lstm_cell_18/addAddV2lstm_cell_18/MatMul:product:0lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_18/addД
#lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_18_biasadd_readvariableop_resource*
_output_shapes	
:Ќ*
dtype02%
#lstm_cell_18/BiasAdd/ReadVariableOp­
lstm_cell_18/BiasAddBiasAddlstm_cell_18/add:z:0+lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_18/BiasAddj
lstm_cell_18/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_18/Const~
lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_18/split/split_dimѓ
lstm_cell_18/splitSplit%lstm_cell_18/split/split_dim:output:0lstm_cell_18/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*
	num_split2
lstm_cell_18/split
lstm_cell_18/SigmoidSigmoidlstm_cell_18/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/Sigmoid
lstm_cell_18/Sigmoid_1Sigmoidlstm_cell_18/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/Sigmoid_1
lstm_cell_18/mulMullstm_cell_18/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/mul}
lstm_cell_18/ReluRelulstm_cell_18/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/Relu
lstm_cell_18/mul_1Mullstm_cell_18/Sigmoid:y:0lstm_cell_18/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/mul_1
lstm_cell_18/add_1AddV2lstm_cell_18/mul:z:0lstm_cell_18/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/add_1
lstm_cell_18/Sigmoid_2Sigmoidlstm_cell_18/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/Sigmoid_2|
lstm_cell_18/Relu_1Relulstm_cell_18/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/Relu_1 
lstm_cell_18/mul_2Mullstm_cell_18/Sigmoid_2:y:0!lstm_cell_18/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterђ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_18_matmul_readvariableop_resource-lstm_cell_18_matmul_1_readvariableop_resource,lstm_cell_18_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџK:џџџџџџџџџK: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_44730166*
condR
while_cond_44730165*K
output_shapes:
8: : : : :џџџџџџџџџK:џџџџџџџџџK: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџK*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeъ
IdentityIdentitytranspose_1:y:0$^lstm_cell_18/BiasAdd/ReadVariableOp#^lstm_cell_18/MatMul/ReadVariableOp%^lstm_cell_18/MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2J
#lstm_cell_18/BiasAdd/ReadVariableOp#lstm_cell_18/BiasAdd/ReadVariableOp2H
"lstm_cell_18/MatMul/ReadVariableOp"lstm_cell_18/MatMul/ReadVariableOp2L
$lstm_cell_18/MatMul_1/ReadVariableOp$lstm_cell_18/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
№	
п
F__inference_dense_27_layer_call_and_return_conditional_losses_44731827

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2 *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
к
Д
while_cond_44726617
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_44726617___redundant_placeholder06
2while_while_cond_44726617___redundant_placeholder16
2while_while_cond_44726617___redundant_placeholder26
2while_while_cond_44726617___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
аP
л
gru_9_while_body_44728942(
$gru_9_while_gru_9_while_loop_counter.
*gru_9_while_gru_9_while_maximum_iterations
gru_9_while_placeholder
gru_9_while_placeholder_1
gru_9_while_placeholder_2'
#gru_9_while_gru_9_strided_slice_1_0c
_gru_9_while_tensorarrayv2read_tensorlistgetitem_gru_9_tensorarrayunstack_tensorlistfromtensor_04
0gru_9_while_gru_cell_9_readvariableop_resource_0;
7gru_9_while_gru_cell_9_matmul_readvariableop_resource_0=
9gru_9_while_gru_cell_9_matmul_1_readvariableop_resource_0
gru_9_while_identity
gru_9_while_identity_1
gru_9_while_identity_2
gru_9_while_identity_3
gru_9_while_identity_4%
!gru_9_while_gru_9_strided_slice_1a
]gru_9_while_tensorarrayv2read_tensorlistgetitem_gru_9_tensorarrayunstack_tensorlistfromtensor2
.gru_9_while_gru_cell_9_readvariableop_resource9
5gru_9_while_gru_cell_9_matmul_readvariableop_resource;
7gru_9_while_gru_cell_9_matmul_1_readvariableop_resourceЂ,gru_9/while/gru_cell_9/MatMul/ReadVariableOpЂ.gru_9/while/gru_cell_9/MatMul_1/ReadVariableOpЂ%gru_9/while/gru_cell_9/ReadVariableOpЯ
=gru_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2?
=gru_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeї
/gru_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_9_while_tensorarrayv2read_tensorlistgetitem_gru_9_tensorarrayunstack_tensorlistfromtensor_0gru_9_while_placeholderFgru_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype021
/gru_9/while/TensorArrayV2Read/TensorListGetItemР
%gru_9/while/gru_cell_9/ReadVariableOpReadVariableOp0gru_9_while_gru_cell_9_readvariableop_resource_0*
_output_shapes
:	*
dtype02'
%gru_9/while/gru_cell_9/ReadVariableOpБ
gru_9/while/gru_cell_9/unstackUnpack-gru_9/while/gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2 
gru_9/while/gru_cell_9/unstackе
,gru_9/while/gru_cell_9/MatMul/ReadVariableOpReadVariableOp7gru_9_while_gru_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02.
,gru_9/while/gru_cell_9/MatMul/ReadVariableOpщ
gru_9/while/gru_cell_9/MatMulMatMul6gru_9/while/TensorArrayV2Read/TensorListGetItem:item:04gru_9/while/gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_9/while/gru_cell_9/MatMulа
gru_9/while/gru_cell_9/BiasAddBiasAdd'gru_9/while/gru_cell_9/MatMul:product:0'gru_9/while/gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2 
gru_9/while/gru_cell_9/BiasAdd~
gru_9/while/gru_cell_9/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_9/while/gru_cell_9/Const
&gru_9/while/gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2(
&gru_9/while/gru_cell_9/split/split_dim
gru_9/while/gru_cell_9/splitSplit/gru_9/while/gru_cell_9/split/split_dim:output:0'gru_9/while/gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
gru_9/while/gru_cell_9/splitл
.gru_9/while/gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp9gru_9_while_gru_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes
:	2*
dtype020
.gru_9/while/gru_cell_9/MatMul_1/ReadVariableOpв
gru_9/while/gru_cell_9/MatMul_1MatMulgru_9_while_placeholder_26gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2!
gru_9/while/gru_cell_9/MatMul_1ж
 gru_9/while/gru_cell_9/BiasAdd_1BiasAdd)gru_9/while/gru_cell_9/MatMul_1:product:0'gru_9/while/gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2"
 gru_9/while/gru_cell_9/BiasAdd_1
gru_9/while/gru_cell_9/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   џџџџ2 
gru_9/while/gru_cell_9/Const_1
(gru_9/while/gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2*
(gru_9/while/gru_cell_9/split_1/split_dimЦ
gru_9/while/gru_cell_9/split_1SplitV)gru_9/while/gru_cell_9/BiasAdd_1:output:0'gru_9/while/gru_cell_9/Const_1:output:01gru_9/while/gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2 
gru_9/while/gru_cell_9/split_1У
gru_9/while/gru_cell_9/addAddV2%gru_9/while/gru_cell_9/split:output:0'gru_9/while/gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_9/while/gru_cell_9/add
gru_9/while/gru_cell_9/SigmoidSigmoidgru_9/while/gru_cell_9/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22 
gru_9/while/gru_cell_9/SigmoidЧ
gru_9/while/gru_cell_9/add_1AddV2%gru_9/while/gru_cell_9/split:output:1'gru_9/while/gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_9/while/gru_cell_9/add_1Ѓ
 gru_9/while/gru_cell_9/Sigmoid_1Sigmoid gru_9/while/gru_cell_9/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22"
 gru_9/while/gru_cell_9/Sigmoid_1Р
gru_9/while/gru_cell_9/mulMul$gru_9/while/gru_cell_9/Sigmoid_1:y:0'gru_9/while/gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_9/while/gru_cell_9/mulО
gru_9/while/gru_cell_9/add_2AddV2%gru_9/while/gru_cell_9/split:output:2gru_9/while/gru_cell_9/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_9/while/gru_cell_9/add_2
gru_9/while/gru_cell_9/ReluRelu gru_9/while/gru_cell_9/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_9/while/gru_cell_9/ReluД
gru_9/while/gru_cell_9/mul_1Mul"gru_9/while/gru_cell_9/Sigmoid:y:0gru_9_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_9/while/gru_cell_9/mul_1
gru_9/while/gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_9/while/gru_cell_9/sub/xМ
gru_9/while/gru_cell_9/subSub%gru_9/while/gru_cell_9/sub/x:output:0"gru_9/while/gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_9/while/gru_cell_9/subР
gru_9/while/gru_cell_9/mul_2Mulgru_9/while/gru_cell_9/sub:z:0)gru_9/while/gru_cell_9/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_9/while/gru_cell_9/mul_2Л
gru_9/while/gru_cell_9/add_3AddV2 gru_9/while/gru_cell_9/mul_1:z:0 gru_9/while/gru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_9/while/gru_cell_9/add_3ќ
0gru_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_9_while_placeholder_1gru_9_while_placeholder gru_9/while/gru_cell_9/add_3:z:0*
_output_shapes
: *
element_dtype022
0gru_9/while/TensorArrayV2Write/TensorListSetItemh
gru_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_9/while/add/y
gru_9/while/addAddV2gru_9_while_placeholdergru_9/while/add/y:output:0*
T0*
_output_shapes
: 2
gru_9/while/addl
gru_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_9/while/add_1/y
gru_9/while/add_1AddV2$gru_9_while_gru_9_while_loop_countergru_9/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru_9/while/add_1ј
gru_9/while/IdentityIdentitygru_9/while/add_1:z:0-^gru_9/while/gru_cell_9/MatMul/ReadVariableOp/^gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp&^gru_9/while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
gru_9/while/Identity
gru_9/while/Identity_1Identity*gru_9_while_gru_9_while_maximum_iterations-^gru_9/while/gru_cell_9/MatMul/ReadVariableOp/^gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp&^gru_9/while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
gru_9/while/Identity_1њ
gru_9/while/Identity_2Identitygru_9/while/add:z:0-^gru_9/while/gru_cell_9/MatMul/ReadVariableOp/^gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp&^gru_9/while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
gru_9/while/Identity_2Ї
gru_9/while/Identity_3Identity@gru_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0-^gru_9/while/gru_cell_9/MatMul/ReadVariableOp/^gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp&^gru_9/while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
gru_9/while/Identity_3
gru_9/while/Identity_4Identity gru_9/while/gru_cell_9/add_3:z:0-^gru_9/while/gru_cell_9/MatMul/ReadVariableOp/^gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp&^gru_9/while/gru_cell_9/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_9/while/Identity_4"H
!gru_9_while_gru_9_strided_slice_1#gru_9_while_gru_9_strided_slice_1_0"t
7gru_9_while_gru_cell_9_matmul_1_readvariableop_resource9gru_9_while_gru_cell_9_matmul_1_readvariableop_resource_0"p
5gru_9_while_gru_cell_9_matmul_readvariableop_resource7gru_9_while_gru_cell_9_matmul_readvariableop_resource_0"b
.gru_9_while_gru_cell_9_readvariableop_resource0gru_9_while_gru_cell_9_readvariableop_resource_0"5
gru_9_while_identitygru_9/while/Identity:output:0"9
gru_9_while_identity_1gru_9/while/Identity_1:output:0"9
gru_9_while_identity_2gru_9/while/Identity_2:output:0"9
gru_9_while_identity_3gru_9/while/Identity_3:output:0"9
gru_9_while_identity_4gru_9/while/Identity_4:output:0"Р
]gru_9_while_tensorarrayv2read_tensorlistgetitem_gru_9_tensorarrayunstack_tensorlistfromtensor_gru_9_while_tensorarrayv2read_tensorlistgetitem_gru_9_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ2: : :::2\
,gru_9/while/gru_cell_9/MatMul/ReadVariableOp,gru_9/while/gru_cell_9/MatMul/ReadVariableOp2`
.gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp.gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp2N
%gru_9/while/gru_cell_9/ReadVariableOp%gru_9/while/gru_cell_9/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
№	
п
F__inference_dense_27_layer_call_and_return_conditional_losses_44728393

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2 *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
Т
Я
/__inference_lstm_cell_19_layer_call_fn_44732196

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2ЂStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_lstm_cell_19_layer_call_and_return_conditional_losses_447267972
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:џџџџџџџџџK:џџџџџџџџџ2:џџџџџџџџџ2:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџK
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ2
"
_user_specified_name
states/0:QM
'
_output_shapes
:џџџџџџџџџ2
"
_user_specified_name
states/1
Є

Ц
*__inference_model_9_layer_call_fn_44729770

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13
identityЂStatefulPartitionedCallЅ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*1
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_model_9_layer_call_and_return_conditional_losses_447286462
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:џџџџџџџџџџџџџџџџџџ:::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ї!
т
while_body_44726618
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_gru_cell_9_44726640_0
while_gru_cell_9_44726642_0
while_gru_cell_9_44726644_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_gru_cell_9_44726640
while_gru_cell_9_44726642
while_gru_cell_9_44726644Ђ(while/gru_cell_9/StatefulPartitionedCallУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЖ
(while/gru_cell_9/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_9_44726640_0while_gru_cell_9_44726642_0while_gru_cell_9_44726644_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ2:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_gru_cell_9_layer_call_and_return_conditional_losses_447262412*
(while/gru_cell_9/StatefulPartitionedCallѕ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/gru_cell_9/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0)^while/gru_cell_9/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/gru_cell_9/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/gru_cell_9/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2И
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/gru_cell_9/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Р
while/Identity_4Identity1while/gru_cell_9/StatefulPartitionedCall:output:1)^while/gru_cell_9/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22
while/Identity_4"8
while_gru_cell_9_44726640while_gru_cell_9_44726640_0"8
while_gru_cell_9_44726642while_gru_cell_9_44726642_0"8
while_gru_cell_9_44726644while_gru_cell_9_44726644_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ2: : :::2T
(while/gru_cell_9/StatefulPartitionedCall(while/gru_cell_9/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
Њ

Ш
*__inference_model_9_layer_call_fn_44728679
input_10
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13
identityЂStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallinput_10unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*1
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_model_9_layer_call_and_return_conditional_losses_447286462
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:џџџџџџџџџџџџџџџџџџ:::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
input_10
Ђ

*__inference_lstm_19_layer_call_fn_44731461
inputs_0
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_lstm_19_layer_call_and_return_conditional_losses_447272922
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџK:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK
"
_user_specified_name
inputs/0


(__inference_gru_9_layer_call_fn_44730793

inputs
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_gru_9_layer_call_and_return_conditional_losses_447279572
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ђ<
к
C__inference_gru_9_layer_call_and_return_conditional_losses_44726564

inputs
gru_cell_9_44726488
gru_cell_9_44726490
gru_cell_9_44726492
identityЂ"gru_cell_9/StatefulPartitionedCallЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2ѕ
"gru_cell_9/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_9_44726488gru_cell_9_44726490gru_cell_9_44726492*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ2:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_gru_cell_9_layer_call_and_return_conditional_losses_447262012$
"gru_cell_9/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterю
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_9_44726488gru_cell_9_44726490gru_cell_9_44726492*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_44726500*
condR
while_cond_44726499*8
output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime
IdentityIdentitystrided_slice_3:output:0#^gru_cell_9/StatefulPartitionedCall^while*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2H
"gru_cell_9/StatefulPartitionedCall"gru_cell_9/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ќD
ф
E__inference_lstm_18_layer_call_and_return_conditional_losses_44726120

inputs
lstm_cell_18_44726038
lstm_cell_18_44726040
lstm_cell_18_44726042
identityЂ$lstm_cell_18/StatefulPartitionedCallЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2Ј
$lstm_cell_18/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_18_44726038lstm_cell_18_44726040lstm_cell_18_44726042*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_lstm_cell_18_layer_call_and_return_conditional_losses_447256252&
$lstm_cell_18/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter­
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_18_44726038lstm_cell_18_44726040lstm_cell_18_44726042*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџK:џџџџџџџџџK: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_44726051*
condR
while_cond_44726050*K
output_shapes:
8: : : : :џџџџџџџџџK:џџџџџџџџџK: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџK*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime
IdentityIdentitytranspose_1:y:0%^lstm_cell_18/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2L
$lstm_cell_18/StatefulPartitionedCall$lstm_cell_18/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
к
Д
while_cond_44730680
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_44730680___redundant_placeholder06
2while_while_cond_44730680___redundant_placeholder16
2while_while_cond_44730680___redundant_placeholder26
2while_while_cond_44730680___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
Љ
f
-__inference_dropout_19_layer_call_fn_44731811

inputs
identityЂStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dropout_19_layer_call_and_return_conditional_losses_447280292
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ222
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
к
Д
while_cond_44730521
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_44730521___redundant_placeholder06
2while_while_cond_44730521___redundant_placeholder16
2while_while_cond_44730521___redundant_placeholder26
2while_while_cond_44730521___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
Щ[
є
E__inference_lstm_18_layer_call_and_return_conditional_losses_44727457

inputs/
+lstm_cell_18_matmul_readvariableop_resource1
-lstm_cell_18_matmul_1_readvariableop_resource0
,lstm_cell_18_biasadd_readvariableop_resource
identityЂ#lstm_cell_18/BiasAdd/ReadVariableOpЂ"lstm_cell_18/MatMul/ReadVariableOpЂ$lstm_cell_18/MatMul_1/ReadVariableOpЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2Е
"lstm_cell_18/MatMul/ReadVariableOpReadVariableOp+lstm_cell_18_matmul_readvariableop_resource*
_output_shapes
:	Ќ*
dtype02$
"lstm_cell_18/MatMul/ReadVariableOp­
lstm_cell_18/MatMulMatMulstrided_slice_2:output:0*lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_18/MatMulЛ
$lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_18_matmul_1_readvariableop_resource*
_output_shapes
:	KЌ*
dtype02&
$lstm_cell_18/MatMul_1/ReadVariableOpЉ
lstm_cell_18/MatMul_1MatMulzeros:output:0,lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_18/MatMul_1 
lstm_cell_18/addAddV2lstm_cell_18/MatMul:product:0lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_18/addД
#lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_18_biasadd_readvariableop_resource*
_output_shapes	
:Ќ*
dtype02%
#lstm_cell_18/BiasAdd/ReadVariableOp­
lstm_cell_18/BiasAddBiasAddlstm_cell_18/add:z:0+lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_18/BiasAddj
lstm_cell_18/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_18/Const~
lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_18/split/split_dimѓ
lstm_cell_18/splitSplit%lstm_cell_18/split/split_dim:output:0lstm_cell_18/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*
	num_split2
lstm_cell_18/split
lstm_cell_18/SigmoidSigmoidlstm_cell_18/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/Sigmoid
lstm_cell_18/Sigmoid_1Sigmoidlstm_cell_18/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/Sigmoid_1
lstm_cell_18/mulMullstm_cell_18/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/mul}
lstm_cell_18/ReluRelulstm_cell_18/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/Relu
lstm_cell_18/mul_1Mullstm_cell_18/Sigmoid:y:0lstm_cell_18/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/mul_1
lstm_cell_18/add_1AddV2lstm_cell_18/mul:z:0lstm_cell_18/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/add_1
lstm_cell_18/Sigmoid_2Sigmoidlstm_cell_18/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/Sigmoid_2|
lstm_cell_18/Relu_1Relulstm_cell_18/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/Relu_1 
lstm_cell_18/mul_2Mullstm_cell_18/Sigmoid_2:y:0!lstm_cell_18/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterђ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_18_matmul_readvariableop_resource-lstm_cell_18_matmul_1_readvariableop_resource,lstm_cell_18_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџK:џџџџџџџџџK: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_44727372*
condR
while_cond_44727371*K
output_shapes:
8: : : : :џџџџџџџџџK:џџџџџџџџџK: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџK*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeъ
IdentityIdentitytranspose_1:y:0$^lstm_cell_18/BiasAdd/ReadVariableOp#^lstm_cell_18/MatMul/ReadVariableOp%^lstm_cell_18/MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2J
#lstm_cell_18/BiasAdd/ReadVariableOp#lstm_cell_18/BiasAdd/ReadVariableOp2H
"lstm_cell_18/MatMul/ReadVariableOp"lstm_cell_18/MatMul/ReadVariableOp2L
$lstm_cell_18/MatMul_1/ReadVariableOp$lstm_cell_18/MatMul_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Њ

Ш
*__inference_model_9_layer_call_fn_44728601
input_10
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13
identityЂStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallinput_10unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*1
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_model_9_layer_call_and_return_conditional_losses_447285682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:џџџџџџџџџџџџџџџџџџ:::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
input_10
Е
Э
while_cond_44730165
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_44730165___redundant_placeholder06
2while_while_cond_44730165___redundant_placeholder16
2while_while_cond_44730165___redundant_placeholder26
2while_while_cond_44730165___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџK:џџџџџџџџџK: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџK:-)
'
_output_shapes
:џџџџџџџџџK:

_output_shapes
: :

_output_shapes
:
І

э
lstm_18_while_cond_44728791,
(lstm_18_while_lstm_18_while_loop_counter2
.lstm_18_while_lstm_18_while_maximum_iterations
lstm_18_while_placeholder
lstm_18_while_placeholder_1
lstm_18_while_placeholder_2
lstm_18_while_placeholder_3.
*lstm_18_while_less_lstm_18_strided_slice_1F
Blstm_18_while_lstm_18_while_cond_44728791___redundant_placeholder0F
Blstm_18_while_lstm_18_while_cond_44728791___redundant_placeholder1F
Blstm_18_while_lstm_18_while_cond_44728791___redundant_placeholder2F
Blstm_18_while_lstm_18_while_cond_44728791___redundant_placeholder3
lstm_18_while_identity

lstm_18/while/LessLesslstm_18_while_placeholder*lstm_18_while_less_lstm_18_strided_slice_1*
T0*
_output_shapes
: 2
lstm_18/while/Lessu
lstm_18/while/IdentityIdentitylstm_18/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_18/while/Identity"9
lstm_18_while_identitylstm_18/while/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџK:џџџџџџџџџK: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџK:-)
'
_output_shapes
:џџџџџџџџџK:

_output_shapes
: :

_output_shapes
:
щZ
ж
C__inference_gru_9_layer_call_and_return_conditional_losses_44730612

inputs&
"gru_cell_9_readvariableop_resource-
)gru_cell_9_matmul_readvariableop_resource/
+gru_cell_9_matmul_1_readvariableop_resource
identityЂ gru_cell_9/MatMul/ReadVariableOpЂ"gru_cell_9/MatMul_1/ReadVariableOpЂgru_cell_9/ReadVariableOpЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2
gru_cell_9/ReadVariableOpReadVariableOp"gru_cell_9_readvariableop_resource*
_output_shapes
:	*
dtype02
gru_cell_9/ReadVariableOp
gru_cell_9/unstackUnpack!gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
gru_cell_9/unstackЏ
 gru_cell_9/MatMul/ReadVariableOpReadVariableOp)gru_cell_9_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 gru_cell_9/MatMul/ReadVariableOpЇ
gru_cell_9/MatMulMatMulstrided_slice_2:output:0(gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_cell_9/MatMul 
gru_cell_9/BiasAddBiasAddgru_cell_9/MatMul:product:0gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_cell_9/BiasAddf
gru_cell_9/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_9/Const
gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_9/split/split_dimи
gru_cell_9/splitSplit#gru_cell_9/split/split_dim:output:0gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
gru_cell_9/splitЕ
"gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_9_matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype02$
"gru_cell_9/MatMul_1/ReadVariableOpЃ
gru_cell_9/MatMul_1MatMulzeros:output:0*gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_cell_9/MatMul_1І
gru_cell_9/BiasAdd_1BiasAddgru_cell_9/MatMul_1:product:0gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_cell_9/BiasAdd_1}
gru_cell_9/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   џџџџ2
gru_cell_9/Const_1
gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_9/split_1/split_dim
gru_cell_9/split_1SplitVgru_cell_9/BiasAdd_1:output:0gru_cell_9/Const_1:output:0%gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
gru_cell_9/split_1
gru_cell_9/addAddV2gru_cell_9/split:output:0gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/addy
gru_cell_9/SigmoidSigmoidgru_cell_9/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/Sigmoid
gru_cell_9/add_1AddV2gru_cell_9/split:output:1gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/add_1
gru_cell_9/Sigmoid_1Sigmoidgru_cell_9/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/Sigmoid_1
gru_cell_9/mulMulgru_cell_9/Sigmoid_1:y:0gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/mul
gru_cell_9/add_2AddV2gru_cell_9/split:output:2gru_cell_9/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/add_2r
gru_cell_9/ReluRelugru_cell_9/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/Relu
gru_cell_9/mul_1Mulgru_cell_9/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/mul_1i
gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_9/sub/x
gru_cell_9/subSubgru_cell_9/sub/x:output:0gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/sub
gru_cell_9/mul_2Mulgru_cell_9/sub:z:0gru_cell_9/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/mul_2
gru_cell_9/add_3AddV2gru_cell_9/mul_1:z:0gru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЋ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_9_readvariableop_resource)gru_cell_9_matmul_readvariableop_resource+gru_cell_9_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_44730522*
condR
while_cond_44730521*8
output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeи
IdentityIdentitystrided_slice_3:output:0!^gru_cell_9/MatMul/ReadVariableOp#^gru_cell_9/MatMul_1/ReadVariableOp^gru_cell_9/ReadVariableOp^while*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2D
 gru_cell_9/MatMul/ReadVariableOp gru_cell_9/MatMul/ReadVariableOp2H
"gru_cell_9/MatMul_1/ReadVariableOp"gru_cell_9/MatMul_1/ReadVariableOp26
gru_cell_9/ReadVariableOpgru_cell_9/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
џZ

#model_9_lstm_19_while_body_44725412<
8model_9_lstm_19_while_model_9_lstm_19_while_loop_counterB
>model_9_lstm_19_while_model_9_lstm_19_while_maximum_iterations%
!model_9_lstm_19_while_placeholder'
#model_9_lstm_19_while_placeholder_1'
#model_9_lstm_19_while_placeholder_2'
#model_9_lstm_19_while_placeholder_3;
7model_9_lstm_19_while_model_9_lstm_19_strided_slice_1_0w
smodel_9_lstm_19_while_tensorarrayv2read_tensorlistgetitem_model_9_lstm_19_tensorarrayunstack_tensorlistfromtensor_0G
Cmodel_9_lstm_19_while_lstm_cell_19_matmul_readvariableop_resource_0I
Emodel_9_lstm_19_while_lstm_cell_19_matmul_1_readvariableop_resource_0H
Dmodel_9_lstm_19_while_lstm_cell_19_biasadd_readvariableop_resource_0"
model_9_lstm_19_while_identity$
 model_9_lstm_19_while_identity_1$
 model_9_lstm_19_while_identity_2$
 model_9_lstm_19_while_identity_3$
 model_9_lstm_19_while_identity_4$
 model_9_lstm_19_while_identity_59
5model_9_lstm_19_while_model_9_lstm_19_strided_slice_1u
qmodel_9_lstm_19_while_tensorarrayv2read_tensorlistgetitem_model_9_lstm_19_tensorarrayunstack_tensorlistfromtensorE
Amodel_9_lstm_19_while_lstm_cell_19_matmul_readvariableop_resourceG
Cmodel_9_lstm_19_while_lstm_cell_19_matmul_1_readvariableop_resourceF
Bmodel_9_lstm_19_while_lstm_cell_19_biasadd_readvariableop_resourceЂ9model_9/lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOpЂ8model_9/lstm_19/while/lstm_cell_19/MatMul/ReadVariableOpЂ:model_9/lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOpу
Gmodel_9/lstm_19/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   2I
Gmodel_9/lstm_19/while/TensorArrayV2Read/TensorListGetItem/element_shapeГ
9model_9/lstm_19/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsmodel_9_lstm_19_while_tensorarrayv2read_tensorlistgetitem_model_9_lstm_19_tensorarrayunstack_tensorlistfromtensor_0!model_9_lstm_19_while_placeholderPmodel_9/lstm_19/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџK*
element_dtype02;
9model_9/lstm_19/while/TensorArrayV2Read/TensorListGetItemљ
8model_9/lstm_19/while/lstm_cell_19/MatMul/ReadVariableOpReadVariableOpCmodel_9_lstm_19_while_lstm_cell_19_matmul_readvariableop_resource_0*
_output_shapes
:	KШ*
dtype02:
8model_9/lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp
)model_9/lstm_19/while/lstm_cell_19/MatMulMatMul@model_9/lstm_19/while/TensorArrayV2Read/TensorListGetItem:item:0@model_9/lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2+
)model_9/lstm_19/while/lstm_cell_19/MatMulџ
:model_9/lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOpEmodel_9_lstm_19_while_lstm_cell_19_matmul_1_readvariableop_resource_0*
_output_shapes
:	2Ш*
dtype02<
:model_9/lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOp
+model_9/lstm_19/while/lstm_cell_19/MatMul_1MatMul#model_9_lstm_19_while_placeholder_2Bmodel_9/lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2-
+model_9/lstm_19/while/lstm_cell_19/MatMul_1ј
&model_9/lstm_19/while/lstm_cell_19/addAddV23model_9/lstm_19/while/lstm_cell_19/MatMul:product:05model_9/lstm_19/while/lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2(
&model_9/lstm_19/while/lstm_cell_19/addј
9model_9/lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOpDmodel_9_lstm_19_while_lstm_cell_19_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02;
9model_9/lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOp
*model_9/lstm_19/while/lstm_cell_19/BiasAddBiasAdd*model_9/lstm_19/while/lstm_cell_19/add:z:0Amodel_9/lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2,
*model_9/lstm_19/while/lstm_cell_19/BiasAdd
(model_9/lstm_19/while/lstm_cell_19/ConstConst*
_output_shapes
: *
dtype0*
value	B :2*
(model_9/lstm_19/while/lstm_cell_19/ConstЊ
2model_9/lstm_19/while/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2model_9/lstm_19/while/lstm_cell_19/split/split_dimЫ
(model_9/lstm_19/while/lstm_cell_19/splitSplit;model_9/lstm_19/while/lstm_cell_19/split/split_dim:output:03model_9/lstm_19/while/lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2*
(model_9/lstm_19/while/lstm_cell_19/splitШ
*model_9/lstm_19/while/lstm_cell_19/SigmoidSigmoid1model_9/lstm_19/while/lstm_cell_19/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22,
*model_9/lstm_19/while/lstm_cell_19/SigmoidЬ
,model_9/lstm_19/while/lstm_cell_19/Sigmoid_1Sigmoid1model_9/lstm_19/while/lstm_cell_19/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22.
,model_9/lstm_19/while/lstm_cell_19/Sigmoid_1р
&model_9/lstm_19/while/lstm_cell_19/mulMul0model_9/lstm_19/while/lstm_cell_19/Sigmoid_1:y:0#model_9_lstm_19_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ22(
&model_9/lstm_19/while/lstm_cell_19/mulП
'model_9/lstm_19/while/lstm_cell_19/ReluRelu1model_9/lstm_19/while/lstm_cell_19/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22)
'model_9/lstm_19/while/lstm_cell_19/Reluє
(model_9/lstm_19/while/lstm_cell_19/mul_1Mul.model_9/lstm_19/while/lstm_cell_19/Sigmoid:y:05model_9/lstm_19/while/lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22*
(model_9/lstm_19/while/lstm_cell_19/mul_1щ
(model_9/lstm_19/while/lstm_cell_19/add_1AddV2*model_9/lstm_19/while/lstm_cell_19/mul:z:0,model_9/lstm_19/while/lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22*
(model_9/lstm_19/while/lstm_cell_19/add_1Ь
,model_9/lstm_19/while/lstm_cell_19/Sigmoid_2Sigmoid1model_9/lstm_19/while/lstm_cell_19/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22.
,model_9/lstm_19/while/lstm_cell_19/Sigmoid_2О
)model_9/lstm_19/while/lstm_cell_19/Relu_1Relu,model_9/lstm_19/while/lstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22+
)model_9/lstm_19/while/lstm_cell_19/Relu_1ј
(model_9/lstm_19/while/lstm_cell_19/mul_2Mul0model_9/lstm_19/while/lstm_cell_19/Sigmoid_2:y:07model_9/lstm_19/while/lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22*
(model_9/lstm_19/while/lstm_cell_19/mul_2А
:model_9/lstm_19/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#model_9_lstm_19_while_placeholder_1!model_9_lstm_19_while_placeholder,model_9/lstm_19/while/lstm_cell_19/mul_2:z:0*
_output_shapes
: *
element_dtype02<
:model_9/lstm_19/while/TensorArrayV2Write/TensorListSetItem|
model_9/lstm_19/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
model_9/lstm_19/while/add/yЉ
model_9/lstm_19/while/addAddV2!model_9_lstm_19_while_placeholder$model_9/lstm_19/while/add/y:output:0*
T0*
_output_shapes
: 2
model_9/lstm_19/while/add
model_9/lstm_19/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
model_9/lstm_19/while/add_1/yЦ
model_9/lstm_19/while/add_1AddV28model_9_lstm_19_while_model_9_lstm_19_while_loop_counter&model_9/lstm_19/while/add_1/y:output:0*
T0*
_output_shapes
: 2
model_9/lstm_19/while/add_1Т
model_9/lstm_19/while/IdentityIdentitymodel_9/lstm_19/while/add_1:z:0:^model_9/lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOp9^model_9/lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp;^model_9/lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2 
model_9/lstm_19/while/Identityх
 model_9/lstm_19/while/Identity_1Identity>model_9_lstm_19_while_model_9_lstm_19_while_maximum_iterations:^model_9/lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOp9^model_9/lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp;^model_9/lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2"
 model_9/lstm_19/while/Identity_1Ф
 model_9/lstm_19/while/Identity_2Identitymodel_9/lstm_19/while/add:z:0:^model_9/lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOp9^model_9/lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp;^model_9/lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2"
 model_9/lstm_19/while/Identity_2ё
 model_9/lstm_19/while/Identity_3IdentityJmodel_9/lstm_19/while/TensorArrayV2Write/TensorListSetItem:output_handle:0:^model_9/lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOp9^model_9/lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp;^model_9/lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2"
 model_9/lstm_19/while/Identity_3ф
 model_9/lstm_19/while/Identity_4Identity,model_9/lstm_19/while/lstm_cell_19/mul_2:z:0:^model_9/lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOp9^model_9/lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp;^model_9/lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22"
 model_9/lstm_19/while/Identity_4ф
 model_9/lstm_19/while/Identity_5Identity,model_9/lstm_19/while/lstm_cell_19/add_1:z:0:^model_9/lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOp9^model_9/lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp;^model_9/lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22"
 model_9/lstm_19/while/Identity_5"I
model_9_lstm_19_while_identity'model_9/lstm_19/while/Identity:output:0"M
 model_9_lstm_19_while_identity_1)model_9/lstm_19/while/Identity_1:output:0"M
 model_9_lstm_19_while_identity_2)model_9/lstm_19/while/Identity_2:output:0"M
 model_9_lstm_19_while_identity_3)model_9/lstm_19/while/Identity_3:output:0"M
 model_9_lstm_19_while_identity_4)model_9/lstm_19/while/Identity_4:output:0"M
 model_9_lstm_19_while_identity_5)model_9/lstm_19/while/Identity_5:output:0"
Bmodel_9_lstm_19_while_lstm_cell_19_biasadd_readvariableop_resourceDmodel_9_lstm_19_while_lstm_cell_19_biasadd_readvariableop_resource_0"
Cmodel_9_lstm_19_while_lstm_cell_19_matmul_1_readvariableop_resourceEmodel_9_lstm_19_while_lstm_cell_19_matmul_1_readvariableop_resource_0"
Amodel_9_lstm_19_while_lstm_cell_19_matmul_readvariableop_resourceCmodel_9_lstm_19_while_lstm_cell_19_matmul_readvariableop_resource_0"p
5model_9_lstm_19_while_model_9_lstm_19_strided_slice_17model_9_lstm_19_while_model_9_lstm_19_strided_slice_1_0"ш
qmodel_9_lstm_19_while_tensorarrayv2read_tensorlistgetitem_model_9_lstm_19_tensorarrayunstack_tensorlistfromtensorsmodel_9_lstm_19_while_tensorarrayv2read_tensorlistgetitem_model_9_lstm_19_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : :::2v
9model_9/lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOp9model_9/lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOp2t
8model_9/lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp8model_9/lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp2x
:model_9/lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOp:model_9/lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
к
Д
while_cond_44727707
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_44727707___redundant_placeholder06
2while_while_cond_44727707___redundant_placeholder16
2while_while_cond_44727707___redundant_placeholder26
2while_while_cond_44727707___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
ф

+__inference_dense_27_layer_call_fn_44731836

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_27_layer_call_and_return_conditional_losses_447283932
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ2::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
Ј
\
0__inference_concatenate_9_layer_call_fn_44731869
inputs_0
inputs_1
identityй
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_concatenate_9_layer_call_and_return_conditional_losses_447284432
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:џџџџџџџџџ :џџџџџџџџџ@:Q M
'
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:џџџџџџџџџ@
"
_user_specified_name
inputs/1


Ф
&__inference_signature_wrapper_44728724
input_10
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_10unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*1
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference__wrapped_model_447255192
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:џџџџџџџџџџџџџџџџџџ:::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
input_10
Р%

while_body_44725919
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0!
while_lstm_cell_18_44725943_0!
while_lstm_cell_18_44725945_0!
while_lstm_cell_18_44725947_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_18_44725943
while_lstm_cell_18_44725945
while_lstm_cell_18_44725947Ђ*while/lstm_cell_18/StatefulPartitionedCallУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemь
*while/lstm_cell_18/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_18_44725943_0while_lstm_cell_18_44725945_0while_lstm_cell_18_44725947_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_lstm_cell_18_layer_call_and_return_conditional_losses_447255922,
*while/lstm_cell_18/StatefulPartitionedCallї
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_18/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0+^while/lstm_cell_18/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations+^while/lstm_cell_18/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0+^while/lstm_cell_18/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2К
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0+^while/lstm_cell_18/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Ф
while/Identity_4Identity3while/lstm_cell_18/StatefulPartitionedCall:output:1+^while/lstm_cell_18/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџK2
while/Identity_4Ф
while/Identity_5Identity3while/lstm_cell_18/StatefulPartitionedCall:output:2+^while/lstm_cell_18/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџK2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"<
while_lstm_cell_18_44725943while_lstm_cell_18_44725943_0"<
while_lstm_cell_18_44725945while_lstm_cell_18_44725945_0"<
while_lstm_cell_18_44725947while_lstm_cell_18_44725947_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџK:џџџџџџџџџK: : :::2X
*while/lstm_cell_18/StatefulPartitionedCall*while/lstm_cell_18/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџK:-)
'
_output_shapes
:џџџџџџџџџK:

_output_shapes
: :

_output_shapes
: 
к
Д
while_cond_44726499
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_44726499___redundant_placeholder06
2while_while_cond_44726499___redundant_placeholder16
2while_while_cond_44726499___redundant_placeholder26
2while_while_cond_44726499___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
б[
і
E__inference_lstm_18_layer_call_and_return_conditional_losses_44730404
inputs_0/
+lstm_cell_18_matmul_readvariableop_resource1
-lstm_cell_18_matmul_1_readvariableop_resource0
,lstm_cell_18_biasadd_readvariableop_resource
identityЂ#lstm_cell_18/BiasAdd/ReadVariableOpЂ"lstm_cell_18/MatMul/ReadVariableOpЂ$lstm_cell_18/MatMul_1/ReadVariableOpЂwhileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2Е
"lstm_cell_18/MatMul/ReadVariableOpReadVariableOp+lstm_cell_18_matmul_readvariableop_resource*
_output_shapes
:	Ќ*
dtype02$
"lstm_cell_18/MatMul/ReadVariableOp­
lstm_cell_18/MatMulMatMulstrided_slice_2:output:0*lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_18/MatMulЛ
$lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_18_matmul_1_readvariableop_resource*
_output_shapes
:	KЌ*
dtype02&
$lstm_cell_18/MatMul_1/ReadVariableOpЉ
lstm_cell_18/MatMul_1MatMulzeros:output:0,lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_18/MatMul_1 
lstm_cell_18/addAddV2lstm_cell_18/MatMul:product:0lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_18/addД
#lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_18_biasadd_readvariableop_resource*
_output_shapes	
:Ќ*
dtype02%
#lstm_cell_18/BiasAdd/ReadVariableOp­
lstm_cell_18/BiasAddBiasAddlstm_cell_18/add:z:0+lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
lstm_cell_18/BiasAddj
lstm_cell_18/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_18/Const~
lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_18/split/split_dimѓ
lstm_cell_18/splitSplit%lstm_cell_18/split/split_dim:output:0lstm_cell_18/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*
	num_split2
lstm_cell_18/split
lstm_cell_18/SigmoidSigmoidlstm_cell_18/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/Sigmoid
lstm_cell_18/Sigmoid_1Sigmoidlstm_cell_18/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/Sigmoid_1
lstm_cell_18/mulMullstm_cell_18/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/mul}
lstm_cell_18/ReluRelulstm_cell_18/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/Relu
lstm_cell_18/mul_1Mullstm_cell_18/Sigmoid:y:0lstm_cell_18/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/mul_1
lstm_cell_18/add_1AddV2lstm_cell_18/mul:z:0lstm_cell_18/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/add_1
lstm_cell_18/Sigmoid_2Sigmoidlstm_cell_18/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/Sigmoid_2|
lstm_cell_18/Relu_1Relulstm_cell_18/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/Relu_1 
lstm_cell_18/mul_2Mullstm_cell_18/Sigmoid_2:y:0!lstm_cell_18/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_cell_18/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterђ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_18_matmul_readvariableop_resource-lstm_cell_18_matmul_1_readvariableop_resource,lstm_cell_18_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџK:џџџџџџџџџK: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_44730319*
condR
while_cond_44730318*K
output_shapes:
8: : : : :џџџџџџџџџK:џџџџџџџџџK: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџK*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeъ
IdentityIdentitytranspose_1:y:0$^lstm_cell_18/BiasAdd/ReadVariableOp#^lstm_cell_18/MatMul/ReadVariableOp%^lstm_cell_18/MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2J
#lstm_cell_18/BiasAdd/ReadVariableOp#lstm_cell_18/BiasAdd/ReadVariableOp2H
"lstm_cell_18/MatMul/ReadVariableOp"lstm_cell_18/MatMul/ReadVariableOp2L
$lstm_cell_18/MatMul_1/ReadVariableOp$lstm_cell_18/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
Е
Э
while_cond_44730318
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_44730318___redundant_placeholder06
2while_while_cond_44730318___redundant_placeholder16
2while_while_cond_44730318___redundant_placeholder26
2while_while_cond_44730318___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџK:џџџџџџџџџK: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџK:-)
'
_output_shapes
:џџџџџџџџџK:

_output_shapes
: :

_output_shapes
:
.
Х
E__inference_model_9_layer_call_and_return_conditional_losses_44728568

inputs
lstm_18_44728528
lstm_18_44728530
lstm_18_44728532
gru_9_44728535
gru_9_44728537
gru_9_44728539
lstm_19_44728544
lstm_19_44728546
lstm_19_44728548
dense_27_44728551
dense_27_44728553
dense_28_44728556
dense_28_44728558
dense_29_44728562
dense_29_44728564
identityЂ dense_27/StatefulPartitionedCallЂ dense_28/StatefulPartitionedCallЂ dense_29/StatefulPartitionedCallЂ"dropout_18/StatefulPartitionedCallЂ"dropout_19/StatefulPartitionedCallЂgru_9/StatefulPartitionedCallЂlstm_18/StatefulPartitionedCallЂlstm_19/StatefulPartitionedCallЙ
lstm_18/StatefulPartitionedCallStatefulPartitionedCallinputslstm_18_44728528lstm_18_44728530lstm_18_44728532*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_lstm_18_layer_call_and_return_conditional_losses_447274572!
lstm_18/StatefulPartitionedCall 
gru_9/StatefulPartitionedCallStatefulPartitionedCallinputsgru_9_44728535gru_9_44728537gru_9_44728539*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_gru_9_layer_call_and_return_conditional_losses_447277982
gru_9/StatefulPartitionedCallІ
"dropout_18/StatefulPartitionedCallStatefulPartitionedCall(lstm_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dropout_18_layer_call_and_return_conditional_losses_447279992$
"dropout_18/StatefulPartitionedCallМ
"dropout_19/StatefulPartitionedCallStatefulPartitionedCall&gru_9/StatefulPartitionedCall:output:0#^dropout_18/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dropout_19_layer_call_and_return_conditional_losses_447280292$
"dropout_19/StatefulPartitionedCallб
lstm_19/StatefulPartitionedCallStatefulPartitionedCall+dropout_18/StatefulPartitionedCall:output:0lstm_19_44728544lstm_19_44728546lstm_19_44728548*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_lstm_19_layer_call_and_return_conditional_losses_447281992!
lstm_19/StatefulPartitionedCallП
 dense_27/StatefulPartitionedCallStatefulPartitionedCall(lstm_19/StatefulPartitionedCall:output:0dense_27_44728551dense_27_44728553*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_27_layer_call_and_return_conditional_losses_447283932"
 dense_27/StatefulPartitionedCallТ
 dense_28/StatefulPartitionedCallStatefulPartitionedCall+dropout_19/StatefulPartitionedCall:output:0dense_28_44728556dense_28_44728558*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_28_layer_call_and_return_conditional_losses_447284202"
 dense_28/StatefulPartitionedCallЗ
concatenate_9/PartitionedCallPartitionedCall)dense_27/StatefulPartitionedCall:output:0)dense_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_concatenate_9_layer_call_and_return_conditional_losses_447284432
concatenate_9/PartitionedCallН
 dense_29/StatefulPartitionedCallStatefulPartitionedCall&concatenate_9/PartitionedCall:output:0dense_29_44728562dense_29_44728564*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_29_layer_call_and_return_conditional_losses_447284622"
 dense_29/StatefulPartitionedCall
IdentityIdentity)dense_29/StatefulPartitionedCall:output:0!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall#^dropout_18/StatefulPartitionedCall#^dropout_19/StatefulPartitionedCall^gru_9/StatefulPartitionedCall ^lstm_18/StatefulPartitionedCall ^lstm_19/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:џџџџџџџџџџџџџџџџџџ:::::::::::::::2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2H
"dropout_18/StatefulPartitionedCall"dropout_18/StatefulPartitionedCall2H
"dropout_19/StatefulPartitionedCall"dropout_19/StatefulPartitionedCall2>
gru_9/StatefulPartitionedCallgru_9/StatefulPartitionedCall2B
lstm_18/StatefulPartitionedCalllstm_18/StatefulPartitionedCall2B
lstm_19/StatefulPartitionedCalllstm_19/StatefulPartitionedCall:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Е
Э
while_cond_44727222
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_44727222___redundant_placeholder06
2while_while_cond_44727222___redundant_placeholder16
2while_while_cond_44727222___redundant_placeholder26
2while_while_cond_44727222___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџ2:џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
г
Џ
H__inference_gru_cell_9_layer_call_and_return_conditional_losses_44732028

inputs
states_0
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1ЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split/split_dimЌ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   џџџџ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split_1/split_dimг
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
Relu^
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:џџџџџџџџџ22
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
subd
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
add_3 
IdentityIdentity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22

IdentityЄ

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ2:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ2
"
_user_specified_name
states/0
Л]
Ы

!model_9_gru_9_while_body_447252568
4model_9_gru_9_while_model_9_gru_9_while_loop_counter>
:model_9_gru_9_while_model_9_gru_9_while_maximum_iterations#
model_9_gru_9_while_placeholder%
!model_9_gru_9_while_placeholder_1%
!model_9_gru_9_while_placeholder_27
3model_9_gru_9_while_model_9_gru_9_strided_slice_1_0s
omodel_9_gru_9_while_tensorarrayv2read_tensorlistgetitem_model_9_gru_9_tensorarrayunstack_tensorlistfromtensor_0<
8model_9_gru_9_while_gru_cell_9_readvariableop_resource_0C
?model_9_gru_9_while_gru_cell_9_matmul_readvariableop_resource_0E
Amodel_9_gru_9_while_gru_cell_9_matmul_1_readvariableop_resource_0 
model_9_gru_9_while_identity"
model_9_gru_9_while_identity_1"
model_9_gru_9_while_identity_2"
model_9_gru_9_while_identity_3"
model_9_gru_9_while_identity_45
1model_9_gru_9_while_model_9_gru_9_strided_slice_1q
mmodel_9_gru_9_while_tensorarrayv2read_tensorlistgetitem_model_9_gru_9_tensorarrayunstack_tensorlistfromtensor:
6model_9_gru_9_while_gru_cell_9_readvariableop_resourceA
=model_9_gru_9_while_gru_cell_9_matmul_readvariableop_resourceC
?model_9_gru_9_while_gru_cell_9_matmul_1_readvariableop_resourceЂ4model_9/gru_9/while/gru_cell_9/MatMul/ReadVariableOpЂ6model_9/gru_9/while/gru_cell_9/MatMul_1/ReadVariableOpЂ-model_9/gru_9/while/gru_cell_9/ReadVariableOpп
Emodel_9/gru_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2G
Emodel_9/gru_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeЇ
7model_9/gru_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemomodel_9_gru_9_while_tensorarrayv2read_tensorlistgetitem_model_9_gru_9_tensorarrayunstack_tensorlistfromtensor_0model_9_gru_9_while_placeholderNmodel_9/gru_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype029
7model_9/gru_9/while/TensorArrayV2Read/TensorListGetItemи
-model_9/gru_9/while/gru_cell_9/ReadVariableOpReadVariableOp8model_9_gru_9_while_gru_cell_9_readvariableop_resource_0*
_output_shapes
:	*
dtype02/
-model_9/gru_9/while/gru_cell_9/ReadVariableOpЩ
&model_9/gru_9/while/gru_cell_9/unstackUnpack5model_9/gru_9/while/gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2(
&model_9/gru_9/while/gru_cell_9/unstackэ
4model_9/gru_9/while/gru_cell_9/MatMul/ReadVariableOpReadVariableOp?model_9_gru_9_while_gru_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype026
4model_9/gru_9/while/gru_cell_9/MatMul/ReadVariableOp
%model_9/gru_9/while/gru_cell_9/MatMulMatMul>model_9/gru_9/while/TensorArrayV2Read/TensorListGetItem:item:0<model_9/gru_9/while/gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2'
%model_9/gru_9/while/gru_cell_9/MatMul№
&model_9/gru_9/while/gru_cell_9/BiasAddBiasAdd/model_9/gru_9/while/gru_cell_9/MatMul:product:0/model_9/gru_9/while/gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2(
&model_9/gru_9/while/gru_cell_9/BiasAdd
$model_9/gru_9/while/gru_cell_9/ConstConst*
_output_shapes
: *
dtype0*
value	B :2&
$model_9/gru_9/while/gru_cell_9/ConstЋ
.model_9/gru_9/while/gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ20
.model_9/gru_9/while/gru_cell_9/split/split_dimЈ
$model_9/gru_9/while/gru_cell_9/splitSplit7model_9/gru_9/while/gru_cell_9/split/split_dim:output:0/model_9/gru_9/while/gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2&
$model_9/gru_9/while/gru_cell_9/splitѓ
6model_9/gru_9/while/gru_cell_9/MatMul_1/ReadVariableOpReadVariableOpAmodel_9_gru_9_while_gru_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes
:	2*
dtype028
6model_9/gru_9/while/gru_cell_9/MatMul_1/ReadVariableOpђ
'model_9/gru_9/while/gru_cell_9/MatMul_1MatMul!model_9_gru_9_while_placeholder_2>model_9/gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2)
'model_9/gru_9/while/gru_cell_9/MatMul_1і
(model_9/gru_9/while/gru_cell_9/BiasAdd_1BiasAdd1model_9/gru_9/while/gru_cell_9/MatMul_1:product:0/model_9/gru_9/while/gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2*
(model_9/gru_9/while/gru_cell_9/BiasAdd_1Ѕ
&model_9/gru_9/while/gru_cell_9/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   џџџџ2(
&model_9/gru_9/while/gru_cell_9/Const_1Џ
0model_9/gru_9/while/gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ22
0model_9/gru_9/while/gru_cell_9/split_1/split_dimю
&model_9/gru_9/while/gru_cell_9/split_1SplitV1model_9/gru_9/while/gru_cell_9/BiasAdd_1:output:0/model_9/gru_9/while/gru_cell_9/Const_1:output:09model_9/gru_9/while/gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2(
&model_9/gru_9/while/gru_cell_9/split_1у
"model_9/gru_9/while/gru_cell_9/addAddV2-model_9/gru_9/while/gru_cell_9/split:output:0/model_9/gru_9/while/gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"model_9/gru_9/while/gru_cell_9/addЕ
&model_9/gru_9/while/gru_cell_9/SigmoidSigmoid&model_9/gru_9/while/gru_cell_9/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&model_9/gru_9/while/gru_cell_9/Sigmoidч
$model_9/gru_9/while/gru_cell_9/add_1AddV2-model_9/gru_9/while/gru_cell_9/split:output:1/model_9/gru_9/while/gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22&
$model_9/gru_9/while/gru_cell_9/add_1Л
(model_9/gru_9/while/gru_cell_9/Sigmoid_1Sigmoid(model_9/gru_9/while/gru_cell_9/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22*
(model_9/gru_9/while/gru_cell_9/Sigmoid_1р
"model_9/gru_9/while/gru_cell_9/mulMul,model_9/gru_9/while/gru_cell_9/Sigmoid_1:y:0/model_9/gru_9/while/gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22$
"model_9/gru_9/while/gru_cell_9/mulо
$model_9/gru_9/while/gru_cell_9/add_2AddV2-model_9/gru_9/while/gru_cell_9/split:output:2&model_9/gru_9/while/gru_cell_9/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$model_9/gru_9/while/gru_cell_9/add_2Ў
#model_9/gru_9/while/gru_cell_9/ReluRelu(model_9/gru_9/while/gru_cell_9/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22%
#model_9/gru_9/while/gru_cell_9/Reluд
$model_9/gru_9/while/gru_cell_9/mul_1Mul*model_9/gru_9/while/gru_cell_9/Sigmoid:y:0!model_9_gru_9_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ22&
$model_9/gru_9/while/gru_cell_9/mul_1
$model_9/gru_9/while/gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2&
$model_9/gru_9/while/gru_cell_9/sub/xм
"model_9/gru_9/while/gru_cell_9/subSub-model_9/gru_9/while/gru_cell_9/sub/x:output:0*model_9/gru_9/while/gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"model_9/gru_9/while/gru_cell_9/subр
$model_9/gru_9/while/gru_cell_9/mul_2Mul&model_9/gru_9/while/gru_cell_9/sub:z:01model_9/gru_9/while/gru_cell_9/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$model_9/gru_9/while/gru_cell_9/mul_2л
$model_9/gru_9/while/gru_cell_9/add_3AddV2(model_9/gru_9/while/gru_cell_9/mul_1:z:0(model_9/gru_9/while/gru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$model_9/gru_9/while/gru_cell_9/add_3Є
8model_9/gru_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!model_9_gru_9_while_placeholder_1model_9_gru_9_while_placeholder(model_9/gru_9/while/gru_cell_9/add_3:z:0*
_output_shapes
: *
element_dtype02:
8model_9/gru_9/while/TensorArrayV2Write/TensorListSetItemx
model_9/gru_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
model_9/gru_9/while/add/yЁ
model_9/gru_9/while/addAddV2model_9_gru_9_while_placeholder"model_9/gru_9/while/add/y:output:0*
T0*
_output_shapes
: 2
model_9/gru_9/while/add|
model_9/gru_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
model_9/gru_9/while/add_1/yМ
model_9/gru_9/while/add_1AddV24model_9_gru_9_while_model_9_gru_9_while_loop_counter$model_9/gru_9/while/add_1/y:output:0*
T0*
_output_shapes
: 2
model_9/gru_9/while/add_1Ј
model_9/gru_9/while/IdentityIdentitymodel_9/gru_9/while/add_1:z:05^model_9/gru_9/while/gru_cell_9/MatMul/ReadVariableOp7^model_9/gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp.^model_9/gru_9/while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
model_9/gru_9/while/IdentityЩ
model_9/gru_9/while/Identity_1Identity:model_9_gru_9_while_model_9_gru_9_while_maximum_iterations5^model_9/gru_9/while/gru_cell_9/MatMul/ReadVariableOp7^model_9/gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp.^model_9/gru_9/while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2 
model_9/gru_9/while/Identity_1Њ
model_9/gru_9/while/Identity_2Identitymodel_9/gru_9/while/add:z:05^model_9/gru_9/while/gru_cell_9/MatMul/ReadVariableOp7^model_9/gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp.^model_9/gru_9/while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2 
model_9/gru_9/while/Identity_2з
model_9/gru_9/while/Identity_3IdentityHmodel_9/gru_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:05^model_9/gru_9/while/gru_cell_9/MatMul/ReadVariableOp7^model_9/gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp.^model_9/gru_9/while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2 
model_9/gru_9/while/Identity_3Ш
model_9/gru_9/while/Identity_4Identity(model_9/gru_9/while/gru_cell_9/add_3:z:05^model_9/gru_9/while/gru_cell_9/MatMul/ReadVariableOp7^model_9/gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp.^model_9/gru_9/while/gru_cell_9/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22 
model_9/gru_9/while/Identity_4"
?model_9_gru_9_while_gru_cell_9_matmul_1_readvariableop_resourceAmodel_9_gru_9_while_gru_cell_9_matmul_1_readvariableop_resource_0"
=model_9_gru_9_while_gru_cell_9_matmul_readvariableop_resource?model_9_gru_9_while_gru_cell_9_matmul_readvariableop_resource_0"r
6model_9_gru_9_while_gru_cell_9_readvariableop_resource8model_9_gru_9_while_gru_cell_9_readvariableop_resource_0"E
model_9_gru_9_while_identity%model_9/gru_9/while/Identity:output:0"I
model_9_gru_9_while_identity_1'model_9/gru_9/while/Identity_1:output:0"I
model_9_gru_9_while_identity_2'model_9/gru_9/while/Identity_2:output:0"I
model_9_gru_9_while_identity_3'model_9/gru_9/while/Identity_3:output:0"I
model_9_gru_9_while_identity_4'model_9/gru_9/while/Identity_4:output:0"h
1model_9_gru_9_while_model_9_gru_9_strided_slice_13model_9_gru_9_while_model_9_gru_9_strided_slice_1_0"р
mmodel_9_gru_9_while_tensorarrayv2read_tensorlistgetitem_model_9_gru_9_tensorarrayunstack_tensorlistfromtensoromodel_9_gru_9_while_tensorarrayv2read_tensorlistgetitem_model_9_gru_9_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ2: : :::2l
4model_9/gru_9/while/gru_cell_9/MatMul/ReadVariableOp4model_9/gru_9/while/gru_cell_9/MatMul/ReadVariableOp2p
6model_9/gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp6model_9/gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp2^
-model_9/gru_9/while/gru_cell_9/ReadVariableOp-model_9/gru_9/while/gru_cell_9/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
: 
б
I
-__inference_dropout_18_layer_call_fn_44730453

inputs
identityж
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dropout_18_layer_call_and_return_conditional_losses_447280042
PartitionedCally
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :џџџџџџџџџџџџџџџџџџK:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK
 
_user_specified_nameinputs
+
§
E__inference_model_9_layer_call_and_return_conditional_losses_44728522
input_10
lstm_18_44728482
lstm_18_44728484
lstm_18_44728486
gru_9_44728489
gru_9_44728491
gru_9_44728493
lstm_19_44728498
lstm_19_44728500
lstm_19_44728502
dense_27_44728505
dense_27_44728507
dense_28_44728510
dense_28_44728512
dense_29_44728516
dense_29_44728518
identityЂ dense_27/StatefulPartitionedCallЂ dense_28/StatefulPartitionedCallЂ dense_29/StatefulPartitionedCallЂgru_9/StatefulPartitionedCallЂlstm_18/StatefulPartitionedCallЂlstm_19/StatefulPartitionedCallЛ
lstm_18/StatefulPartitionedCallStatefulPartitionedCallinput_10lstm_18_44728482lstm_18_44728484lstm_18_44728486*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_lstm_18_layer_call_and_return_conditional_losses_447276102!
lstm_18/StatefulPartitionedCallЂ
gru_9/StatefulPartitionedCallStatefulPartitionedCallinput_10gru_9_44728489gru_9_44728491gru_9_44728493*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_gru_9_layer_call_and_return_conditional_losses_447279572
gru_9/StatefulPartitionedCall
dropout_18/PartitionedCallPartitionedCall(lstm_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dropout_18_layer_call_and_return_conditional_losses_447280042
dropout_18/PartitionedCallџ
dropout_19/PartitionedCallPartitionedCall&gru_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dropout_19_layer_call_and_return_conditional_losses_447280342
dropout_19/PartitionedCallЩ
lstm_19/StatefulPartitionedCallStatefulPartitionedCall#dropout_18/PartitionedCall:output:0lstm_19_44728498lstm_19_44728500lstm_19_44728502*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_lstm_19_layer_call_and_return_conditional_losses_447283522!
lstm_19/StatefulPartitionedCallП
 dense_27/StatefulPartitionedCallStatefulPartitionedCall(lstm_19/StatefulPartitionedCall:output:0dense_27_44728505dense_27_44728507*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_27_layer_call_and_return_conditional_losses_447283932"
 dense_27/StatefulPartitionedCallК
 dense_28/StatefulPartitionedCallStatefulPartitionedCall#dropout_19/PartitionedCall:output:0dense_28_44728510dense_28_44728512*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_28_layer_call_and_return_conditional_losses_447284202"
 dense_28/StatefulPartitionedCallЗ
concatenate_9/PartitionedCallPartitionedCall)dense_27/StatefulPartitionedCall:output:0)dense_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_concatenate_9_layer_call_and_return_conditional_losses_447284432
concatenate_9/PartitionedCallН
 dense_29/StatefulPartitionedCallStatefulPartitionedCall&concatenate_9/PartitionedCall:output:0dense_29_44728516dense_29_44728518*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_29_layer_call_and_return_conditional_losses_447284622"
 dense_29/StatefulPartitionedCallЪ
IdentityIdentity)dense_29/StatefulPartitionedCall:output:0!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall^gru_9/StatefulPartitionedCall ^lstm_18/StatefulPartitionedCall ^lstm_19/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:џџџџџџџџџџџџџџџџџџ:::::::::::::::2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2>
gru_9/StatefulPartitionedCallgru_9/StatefulPartitionedCall2B
lstm_18/StatefulPartitionedCalllstm_18/StatefulPartitionedCall2B
lstm_19/StatefulPartitionedCalllstm_19/StatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
input_10
Ч
д
#__inference__wrapped_model_44725519
input_10?
;model_9_lstm_18_lstm_cell_18_matmul_readvariableop_resourceA
=model_9_lstm_18_lstm_cell_18_matmul_1_readvariableop_resource@
<model_9_lstm_18_lstm_cell_18_biasadd_readvariableop_resource4
0model_9_gru_9_gru_cell_9_readvariableop_resource;
7model_9_gru_9_gru_cell_9_matmul_readvariableop_resource=
9model_9_gru_9_gru_cell_9_matmul_1_readvariableop_resource?
;model_9_lstm_19_lstm_cell_19_matmul_readvariableop_resourceA
=model_9_lstm_19_lstm_cell_19_matmul_1_readvariableop_resource@
<model_9_lstm_19_lstm_cell_19_biasadd_readvariableop_resource3
/model_9_dense_27_matmul_readvariableop_resource4
0model_9_dense_27_biasadd_readvariableop_resource3
/model_9_dense_28_matmul_readvariableop_resource4
0model_9_dense_28_biasadd_readvariableop_resource3
/model_9_dense_29_matmul_readvariableop_resource4
0model_9_dense_29_biasadd_readvariableop_resource
identityЂ'model_9/dense_27/BiasAdd/ReadVariableOpЂ&model_9/dense_27/MatMul/ReadVariableOpЂ'model_9/dense_28/BiasAdd/ReadVariableOpЂ&model_9/dense_28/MatMul/ReadVariableOpЂ'model_9/dense_29/BiasAdd/ReadVariableOpЂ&model_9/dense_29/MatMul/ReadVariableOpЂ.model_9/gru_9/gru_cell_9/MatMul/ReadVariableOpЂ0model_9/gru_9/gru_cell_9/MatMul_1/ReadVariableOpЂ'model_9/gru_9/gru_cell_9/ReadVariableOpЂmodel_9/gru_9/whileЂ3model_9/lstm_18/lstm_cell_18/BiasAdd/ReadVariableOpЂ2model_9/lstm_18/lstm_cell_18/MatMul/ReadVariableOpЂ4model_9/lstm_18/lstm_cell_18/MatMul_1/ReadVariableOpЂmodel_9/lstm_18/whileЂ3model_9/lstm_19/lstm_cell_19/BiasAdd/ReadVariableOpЂ2model_9/lstm_19/lstm_cell_19/MatMul/ReadVariableOpЂ4model_9/lstm_19/lstm_cell_19/MatMul_1/ReadVariableOpЂmodel_9/lstm_19/whilef
model_9/lstm_18/ShapeShapeinput_10*
T0*
_output_shapes
:2
model_9/lstm_18/Shape
#model_9/lstm_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#model_9/lstm_18/strided_slice/stack
%model_9/lstm_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%model_9/lstm_18/strided_slice/stack_1
%model_9/lstm_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%model_9/lstm_18/strided_slice/stack_2Т
model_9/lstm_18/strided_sliceStridedSlicemodel_9/lstm_18/Shape:output:0,model_9/lstm_18/strided_slice/stack:output:0.model_9/lstm_18/strided_slice/stack_1:output:0.model_9/lstm_18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model_9/lstm_18/strided_slice|
model_9/lstm_18/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
model_9/lstm_18/zeros/mul/yЌ
model_9/lstm_18/zeros/mulMul&model_9/lstm_18/strided_slice:output:0$model_9/lstm_18/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
model_9/lstm_18/zeros/mul
model_9/lstm_18/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
model_9/lstm_18/zeros/Less/yЇ
model_9/lstm_18/zeros/LessLessmodel_9/lstm_18/zeros/mul:z:0%model_9/lstm_18/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
model_9/lstm_18/zeros/Less
model_9/lstm_18/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2 
model_9/lstm_18/zeros/packed/1У
model_9/lstm_18/zeros/packedPack&model_9/lstm_18/strided_slice:output:0'model_9/lstm_18/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
model_9/lstm_18/zeros/packed
model_9/lstm_18/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model_9/lstm_18/zeros/ConstЕ
model_9/lstm_18/zerosFill%model_9/lstm_18/zeros/packed:output:0$model_9/lstm_18/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
model_9/lstm_18/zeros
model_9/lstm_18/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2
model_9/lstm_18/zeros_1/mul/yВ
model_9/lstm_18/zeros_1/mulMul&model_9/lstm_18/strided_slice:output:0&model_9/lstm_18/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
model_9/lstm_18/zeros_1/mul
model_9/lstm_18/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2 
model_9/lstm_18/zeros_1/Less/yЏ
model_9/lstm_18/zeros_1/LessLessmodel_9/lstm_18/zeros_1/mul:z:0'model_9/lstm_18/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
model_9/lstm_18/zeros_1/Less
 model_9/lstm_18/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2"
 model_9/lstm_18/zeros_1/packed/1Щ
model_9/lstm_18/zeros_1/packedPack&model_9/lstm_18/strided_slice:output:0)model_9/lstm_18/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
model_9/lstm_18/zeros_1/packed
model_9/lstm_18/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model_9/lstm_18/zeros_1/ConstН
model_9/lstm_18/zeros_1Fill'model_9/lstm_18/zeros_1/packed:output:0&model_9/lstm_18/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
model_9/lstm_18/zeros_1
model_9/lstm_18/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
model_9/lstm_18/transpose/permЕ
model_9/lstm_18/transpose	Transposeinput_10'model_9/lstm_18/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
model_9/lstm_18/transpose
model_9/lstm_18/Shape_1Shapemodel_9/lstm_18/transpose:y:0*
T0*
_output_shapes
:2
model_9/lstm_18/Shape_1
%model_9/lstm_18/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%model_9/lstm_18/strided_slice_1/stack
'model_9/lstm_18/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'model_9/lstm_18/strided_slice_1/stack_1
'model_9/lstm_18/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'model_9/lstm_18/strided_slice_1/stack_2Ю
model_9/lstm_18/strided_slice_1StridedSlice model_9/lstm_18/Shape_1:output:0.model_9/lstm_18/strided_slice_1/stack:output:00model_9/lstm_18/strided_slice_1/stack_1:output:00model_9/lstm_18/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
model_9/lstm_18/strided_slice_1Ѕ
+model_9/lstm_18/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2-
+model_9/lstm_18/TensorArrayV2/element_shapeђ
model_9/lstm_18/TensorArrayV2TensorListReserve4model_9/lstm_18/TensorArrayV2/element_shape:output:0(model_9/lstm_18/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
model_9/lstm_18/TensorArrayV2п
Emodel_9/lstm_18/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2G
Emodel_9/lstm_18/TensorArrayUnstack/TensorListFromTensor/element_shapeИ
7model_9/lstm_18/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel_9/lstm_18/transpose:y:0Nmodel_9/lstm_18/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7model_9/lstm_18/TensorArrayUnstack/TensorListFromTensor
%model_9/lstm_18/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%model_9/lstm_18/strided_slice_2/stack
'model_9/lstm_18/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'model_9/lstm_18/strided_slice_2/stack_1
'model_9/lstm_18/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'model_9/lstm_18/strided_slice_2/stack_2м
model_9/lstm_18/strided_slice_2StridedSlicemodel_9/lstm_18/transpose:y:0.model_9/lstm_18/strided_slice_2/stack:output:00model_9/lstm_18/strided_slice_2/stack_1:output:00model_9/lstm_18/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2!
model_9/lstm_18/strided_slice_2х
2model_9/lstm_18/lstm_cell_18/MatMul/ReadVariableOpReadVariableOp;model_9_lstm_18_lstm_cell_18_matmul_readvariableop_resource*
_output_shapes
:	Ќ*
dtype024
2model_9/lstm_18/lstm_cell_18/MatMul/ReadVariableOpэ
#model_9/lstm_18/lstm_cell_18/MatMulMatMul(model_9/lstm_18/strided_slice_2:output:0:model_9/lstm_18/lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2%
#model_9/lstm_18/lstm_cell_18/MatMulы
4model_9/lstm_18/lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp=model_9_lstm_18_lstm_cell_18_matmul_1_readvariableop_resource*
_output_shapes
:	KЌ*
dtype026
4model_9/lstm_18/lstm_cell_18/MatMul_1/ReadVariableOpщ
%model_9/lstm_18/lstm_cell_18/MatMul_1MatMulmodel_9/lstm_18/zeros:output:0<model_9/lstm_18/lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2'
%model_9/lstm_18/lstm_cell_18/MatMul_1р
 model_9/lstm_18/lstm_cell_18/addAddV2-model_9/lstm_18/lstm_cell_18/MatMul:product:0/model_9/lstm_18/lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2"
 model_9/lstm_18/lstm_cell_18/addф
3model_9/lstm_18/lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp<model_9_lstm_18_lstm_cell_18_biasadd_readvariableop_resource*
_output_shapes	
:Ќ*
dtype025
3model_9/lstm_18/lstm_cell_18/BiasAdd/ReadVariableOpэ
$model_9/lstm_18/lstm_cell_18/BiasAddBiasAdd$model_9/lstm_18/lstm_cell_18/add:z:0;model_9/lstm_18/lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2&
$model_9/lstm_18/lstm_cell_18/BiasAdd
"model_9/lstm_18/lstm_cell_18/ConstConst*
_output_shapes
: *
dtype0*
value	B :2$
"model_9/lstm_18/lstm_cell_18/Const
,model_9/lstm_18/lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,model_9/lstm_18/lstm_cell_18/split/split_dimГ
"model_9/lstm_18/lstm_cell_18/splitSplit5model_9/lstm_18/lstm_cell_18/split/split_dim:output:0-model_9/lstm_18/lstm_cell_18/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*
	num_split2$
"model_9/lstm_18/lstm_cell_18/splitЖ
$model_9/lstm_18/lstm_cell_18/SigmoidSigmoid+model_9/lstm_18/lstm_cell_18/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2&
$model_9/lstm_18/lstm_cell_18/SigmoidК
&model_9/lstm_18/lstm_cell_18/Sigmoid_1Sigmoid+model_9/lstm_18/lstm_cell_18/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџK2(
&model_9/lstm_18/lstm_cell_18/Sigmoid_1Ы
 model_9/lstm_18/lstm_cell_18/mulMul*model_9/lstm_18/lstm_cell_18/Sigmoid_1:y:0 model_9/lstm_18/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2"
 model_9/lstm_18/lstm_cell_18/mul­
!model_9/lstm_18/lstm_cell_18/ReluRelu+model_9/lstm_18/lstm_cell_18/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџK2#
!model_9/lstm_18/lstm_cell_18/Reluм
"model_9/lstm_18/lstm_cell_18/mul_1Mul(model_9/lstm_18/lstm_cell_18/Sigmoid:y:0/model_9/lstm_18/lstm_cell_18/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2$
"model_9/lstm_18/lstm_cell_18/mul_1б
"model_9/lstm_18/lstm_cell_18/add_1AddV2$model_9/lstm_18/lstm_cell_18/mul:z:0&model_9/lstm_18/lstm_cell_18/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2$
"model_9/lstm_18/lstm_cell_18/add_1К
&model_9/lstm_18/lstm_cell_18/Sigmoid_2Sigmoid+model_9/lstm_18/lstm_cell_18/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџK2(
&model_9/lstm_18/lstm_cell_18/Sigmoid_2Ќ
#model_9/lstm_18/lstm_cell_18/Relu_1Relu&model_9/lstm_18/lstm_cell_18/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2%
#model_9/lstm_18/lstm_cell_18/Relu_1р
"model_9/lstm_18/lstm_cell_18/mul_2Mul*model_9/lstm_18/lstm_cell_18/Sigmoid_2:y:01model_9/lstm_18/lstm_cell_18/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2$
"model_9/lstm_18/lstm_cell_18/mul_2Џ
-model_9/lstm_18/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   2/
-model_9/lstm_18/TensorArrayV2_1/element_shapeј
model_9/lstm_18/TensorArrayV2_1TensorListReserve6model_9/lstm_18/TensorArrayV2_1/element_shape:output:0(model_9/lstm_18/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
model_9/lstm_18/TensorArrayV2_1n
model_9/lstm_18/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
model_9/lstm_18/time
(model_9/lstm_18/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2*
(model_9/lstm_18/while/maximum_iterations
"model_9/lstm_18/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"model_9/lstm_18/while/loop_counterт
model_9/lstm_18/whileWhile+model_9/lstm_18/while/loop_counter:output:01model_9/lstm_18/while/maximum_iterations:output:0model_9/lstm_18/time:output:0(model_9/lstm_18/TensorArrayV2_1:handle:0model_9/lstm_18/zeros:output:0 model_9/lstm_18/zeros_1:output:0(model_9/lstm_18/strided_slice_1:output:0Gmodel_9/lstm_18/TensorArrayUnstack/TensorListFromTensor:output_handle:0;model_9_lstm_18_lstm_cell_18_matmul_readvariableop_resource=model_9_lstm_18_lstm_cell_18_matmul_1_readvariableop_resource<model_9_lstm_18_lstm_cell_18_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџK:џџџџџџџџџK: : : : : *%
_read_only_resource_inputs
	
*/
body'R%
#model_9_lstm_18_while_body_44725106*/
cond'R%
#model_9_lstm_18_while_cond_44725105*K
output_shapes:
8: : : : :џџџџџџџџџK:џџџџџџџџџK: : : : : *
parallel_iterations 2
model_9/lstm_18/whileе
@model_9/lstm_18/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   2B
@model_9/lstm_18/TensorArrayV2Stack/TensorListStack/element_shapeБ
2model_9/lstm_18/TensorArrayV2Stack/TensorListStackTensorListStackmodel_9/lstm_18/while:output:3Imodel_9/lstm_18/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK*
element_dtype024
2model_9/lstm_18/TensorArrayV2Stack/TensorListStackЁ
%model_9/lstm_18/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2'
%model_9/lstm_18/strided_slice_3/stack
'model_9/lstm_18/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'model_9/lstm_18/strided_slice_3/stack_1
'model_9/lstm_18/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'model_9/lstm_18/strided_slice_3/stack_2њ
model_9/lstm_18/strided_slice_3StridedSlice;model_9/lstm_18/TensorArrayV2Stack/TensorListStack:tensor:0.model_9/lstm_18/strided_slice_3/stack:output:00model_9/lstm_18/strided_slice_3/stack_1:output:00model_9/lstm_18/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџK*
shrink_axis_mask2!
model_9/lstm_18/strided_slice_3
 model_9/lstm_18/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 model_9/lstm_18/transpose_1/permю
model_9/lstm_18/transpose_1	Transpose;model_9/lstm_18/TensorArrayV2Stack/TensorListStack:tensor:0)model_9/lstm_18/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
model_9/lstm_18/transpose_1
model_9/lstm_18/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
model_9/lstm_18/runtimeb
model_9/gru_9/ShapeShapeinput_10*
T0*
_output_shapes
:2
model_9/gru_9/Shape
!model_9/gru_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!model_9/gru_9/strided_slice/stack
#model_9/gru_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#model_9/gru_9/strided_slice/stack_1
#model_9/gru_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#model_9/gru_9/strided_slice/stack_2Ж
model_9/gru_9/strided_sliceStridedSlicemodel_9/gru_9/Shape:output:0*model_9/gru_9/strided_slice/stack:output:0,model_9/gru_9/strided_slice/stack_1:output:0,model_9/gru_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model_9/gru_9/strided_slicex
model_9/gru_9/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
model_9/gru_9/zeros/mul/yЄ
model_9/gru_9/zeros/mulMul$model_9/gru_9/strided_slice:output:0"model_9/gru_9/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
model_9/gru_9/zeros/mul{
model_9/gru_9/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
model_9/gru_9/zeros/Less/y
model_9/gru_9/zeros/LessLessmodel_9/gru_9/zeros/mul:z:0#model_9/gru_9/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
model_9/gru_9/zeros/Less~
model_9/gru_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
model_9/gru_9/zeros/packed/1Л
model_9/gru_9/zeros/packedPack$model_9/gru_9/strided_slice:output:0%model_9/gru_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
model_9/gru_9/zeros/packed{
model_9/gru_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model_9/gru_9/zeros/Const­
model_9/gru_9/zerosFill#model_9/gru_9/zeros/packed:output:0"model_9/gru_9/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
model_9/gru_9/zeros
model_9/gru_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
model_9/gru_9/transpose/permЏ
model_9/gru_9/transpose	Transposeinput_10%model_9/gru_9/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
model_9/gru_9/transposey
model_9/gru_9/Shape_1Shapemodel_9/gru_9/transpose:y:0*
T0*
_output_shapes
:2
model_9/gru_9/Shape_1
#model_9/gru_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#model_9/gru_9/strided_slice_1/stack
%model_9/gru_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%model_9/gru_9/strided_slice_1/stack_1
%model_9/gru_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%model_9/gru_9/strided_slice_1/stack_2Т
model_9/gru_9/strided_slice_1StridedSlicemodel_9/gru_9/Shape_1:output:0,model_9/gru_9/strided_slice_1/stack:output:0.model_9/gru_9/strided_slice_1/stack_1:output:0.model_9/gru_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model_9/gru_9/strided_slice_1Ё
)model_9/gru_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)model_9/gru_9/TensorArrayV2/element_shapeъ
model_9/gru_9/TensorArrayV2TensorListReserve2model_9/gru_9/TensorArrayV2/element_shape:output:0&model_9/gru_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
model_9/gru_9/TensorArrayV2л
Cmodel_9/gru_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2E
Cmodel_9/gru_9/TensorArrayUnstack/TensorListFromTensor/element_shapeА
5model_9/gru_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel_9/gru_9/transpose:y:0Lmodel_9/gru_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type027
5model_9/gru_9/TensorArrayUnstack/TensorListFromTensor
#model_9/gru_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#model_9/gru_9/strided_slice_2/stack
%model_9/gru_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%model_9/gru_9/strided_slice_2/stack_1
%model_9/gru_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%model_9/gru_9/strided_slice_2/stack_2а
model_9/gru_9/strided_slice_2StridedSlicemodel_9/gru_9/transpose:y:0,model_9/gru_9/strided_slice_2/stack:output:0.model_9/gru_9/strided_slice_2/stack_1:output:0.model_9/gru_9/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
model_9/gru_9/strided_slice_2Ф
'model_9/gru_9/gru_cell_9/ReadVariableOpReadVariableOp0model_9_gru_9_gru_cell_9_readvariableop_resource*
_output_shapes
:	*
dtype02)
'model_9/gru_9/gru_cell_9/ReadVariableOpЗ
 model_9/gru_9/gru_cell_9/unstackUnpack/model_9/gru_9/gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2"
 model_9/gru_9/gru_cell_9/unstackй
.model_9/gru_9/gru_cell_9/MatMul/ReadVariableOpReadVariableOp7model_9_gru_9_gru_cell_9_matmul_readvariableop_resource*
_output_shapes
:	*
dtype020
.model_9/gru_9/gru_cell_9/MatMul/ReadVariableOpп
model_9/gru_9/gru_cell_9/MatMulMatMul&model_9/gru_9/strided_slice_2:output:06model_9/gru_9/gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2!
model_9/gru_9/gru_cell_9/MatMulи
 model_9/gru_9/gru_cell_9/BiasAddBiasAdd)model_9/gru_9/gru_cell_9/MatMul:product:0)model_9/gru_9/gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2"
 model_9/gru_9/gru_cell_9/BiasAdd
model_9/gru_9/gru_cell_9/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
model_9/gru_9/gru_cell_9/Const
(model_9/gru_9/gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2*
(model_9/gru_9/gru_cell_9/split/split_dim
model_9/gru_9/gru_cell_9/splitSplit1model_9/gru_9/gru_cell_9/split/split_dim:output:0)model_9/gru_9/gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2 
model_9/gru_9/gru_cell_9/splitп
0model_9/gru_9/gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp9model_9_gru_9_gru_cell_9_matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype022
0model_9/gru_9/gru_cell_9/MatMul_1/ReadVariableOpл
!model_9/gru_9/gru_cell_9/MatMul_1MatMulmodel_9/gru_9/zeros:output:08model_9/gru_9/gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2#
!model_9/gru_9/gru_cell_9/MatMul_1о
"model_9/gru_9/gru_cell_9/BiasAdd_1BiasAdd+model_9/gru_9/gru_cell_9/MatMul_1:product:0)model_9/gru_9/gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2$
"model_9/gru_9/gru_cell_9/BiasAdd_1
 model_9/gru_9/gru_cell_9/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   џџџџ2"
 model_9/gru_9/gru_cell_9/Const_1Ѓ
*model_9/gru_9/gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2,
*model_9/gru_9/gru_cell_9/split_1/split_dimа
 model_9/gru_9/gru_cell_9/split_1SplitV+model_9/gru_9/gru_cell_9/BiasAdd_1:output:0)model_9/gru_9/gru_cell_9/Const_1:output:03model_9/gru_9/gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2"
 model_9/gru_9/gru_cell_9/split_1Ы
model_9/gru_9/gru_cell_9/addAddV2'model_9/gru_9/gru_cell_9/split:output:0)model_9/gru_9/gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
model_9/gru_9/gru_cell_9/addЃ
 model_9/gru_9/gru_cell_9/SigmoidSigmoid model_9/gru_9/gru_cell_9/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22"
 model_9/gru_9/gru_cell_9/SigmoidЯ
model_9/gru_9/gru_cell_9/add_1AddV2'model_9/gru_9/gru_cell_9/split:output:1)model_9/gru_9/gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22 
model_9/gru_9/gru_cell_9/add_1Љ
"model_9/gru_9/gru_cell_9/Sigmoid_1Sigmoid"model_9/gru_9/gru_cell_9/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"model_9/gru_9/gru_cell_9/Sigmoid_1Ш
model_9/gru_9/gru_cell_9/mulMul&model_9/gru_9/gru_cell_9/Sigmoid_1:y:0)model_9/gru_9/gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
model_9/gru_9/gru_cell_9/mulЦ
model_9/gru_9/gru_cell_9/add_2AddV2'model_9/gru_9/gru_cell_9/split:output:2 model_9/gru_9/gru_cell_9/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22 
model_9/gru_9/gru_cell_9/add_2
model_9/gru_9/gru_cell_9/ReluRelu"model_9/gru_9/gru_cell_9/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
model_9/gru_9/gru_cell_9/ReluН
model_9/gru_9/gru_cell_9/mul_1Mul$model_9/gru_9/gru_cell_9/Sigmoid:y:0model_9/gru_9/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22 
model_9/gru_9/gru_cell_9/mul_1
model_9/gru_9/gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2 
model_9/gru_9/gru_cell_9/sub/xФ
model_9/gru_9/gru_cell_9/subSub'model_9/gru_9/gru_cell_9/sub/x:output:0$model_9/gru_9/gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
model_9/gru_9/gru_cell_9/subШ
model_9/gru_9/gru_cell_9/mul_2Mul model_9/gru_9/gru_cell_9/sub:z:0+model_9/gru_9/gru_cell_9/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22 
model_9/gru_9/gru_cell_9/mul_2У
model_9/gru_9/gru_cell_9/add_3AddV2"model_9/gru_9/gru_cell_9/mul_1:z:0"model_9/gru_9/gru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22 
model_9/gru_9/gru_cell_9/add_3Ћ
+model_9/gru_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2-
+model_9/gru_9/TensorArrayV2_1/element_shape№
model_9/gru_9/TensorArrayV2_1TensorListReserve4model_9/gru_9/TensorArrayV2_1/element_shape:output:0&model_9/gru_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
model_9/gru_9/TensorArrayV2_1j
model_9/gru_9/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
model_9/gru_9/time
&model_9/gru_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2(
&model_9/gru_9/while/maximum_iterations
 model_9/gru_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2"
 model_9/gru_9/while/loop_counterя
model_9/gru_9/whileWhile)model_9/gru_9/while/loop_counter:output:0/model_9/gru_9/while/maximum_iterations:output:0model_9/gru_9/time:output:0&model_9/gru_9/TensorArrayV2_1:handle:0model_9/gru_9/zeros:output:0&model_9/gru_9/strided_slice_1:output:0Emodel_9/gru_9/TensorArrayUnstack/TensorListFromTensor:output_handle:00model_9_gru_9_gru_cell_9_readvariableop_resource7model_9_gru_9_gru_cell_9_matmul_readvariableop_resource9model_9_gru_9_gru_cell_9_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	*-
body%R#
!model_9_gru_9_while_body_44725256*-
cond%R#
!model_9_gru_9_while_cond_44725255*8
output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *
parallel_iterations 2
model_9/gru_9/whileб
>model_9/gru_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2@
>model_9/gru_9/TensorArrayV2Stack/TensorListStack/element_shapeЉ
0model_9/gru_9/TensorArrayV2Stack/TensorListStackTensorListStackmodel_9/gru_9/while:output:3Gmodel_9/gru_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype022
0model_9/gru_9/TensorArrayV2Stack/TensorListStack
#model_9/gru_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2%
#model_9/gru_9/strided_slice_3/stack
%model_9/gru_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%model_9/gru_9/strided_slice_3/stack_1
%model_9/gru_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%model_9/gru_9/strided_slice_3/stack_2ю
model_9/gru_9/strided_slice_3StridedSlice9model_9/gru_9/TensorArrayV2Stack/TensorListStack:tensor:0,model_9/gru_9/strided_slice_3/stack:output:0.model_9/gru_9/strided_slice_3/stack_1:output:0.model_9/gru_9/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
model_9/gru_9/strided_slice_3
model_9/gru_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
model_9/gru_9/transpose_1/permц
model_9/gru_9/transpose_1	Transpose9model_9/gru_9/TensorArrayV2Stack/TensorListStack:tensor:0'model_9/gru_9/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
model_9/gru_9/transpose_1
model_9/gru_9/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
model_9/gru_9/runtimeІ
model_9/dropout_18/IdentityIdentitymodel_9/lstm_18/transpose_1:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
model_9/dropout_18/Identity 
model_9/dropout_19/IdentityIdentity&model_9/gru_9/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
model_9/dropout_19/Identity
model_9/lstm_19/ShapeShape$model_9/dropout_18/Identity:output:0*
T0*
_output_shapes
:2
model_9/lstm_19/Shape
#model_9/lstm_19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#model_9/lstm_19/strided_slice/stack
%model_9/lstm_19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%model_9/lstm_19/strided_slice/stack_1
%model_9/lstm_19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%model_9/lstm_19/strided_slice/stack_2Т
model_9/lstm_19/strided_sliceStridedSlicemodel_9/lstm_19/Shape:output:0,model_9/lstm_19/strided_slice/stack:output:0.model_9/lstm_19/strided_slice/stack_1:output:0.model_9/lstm_19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model_9/lstm_19/strided_slice|
model_9/lstm_19/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
model_9/lstm_19/zeros/mul/yЌ
model_9/lstm_19/zeros/mulMul&model_9/lstm_19/strided_slice:output:0$model_9/lstm_19/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
model_9/lstm_19/zeros/mul
model_9/lstm_19/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
model_9/lstm_19/zeros/Less/yЇ
model_9/lstm_19/zeros/LessLessmodel_9/lstm_19/zeros/mul:z:0%model_9/lstm_19/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
model_9/lstm_19/zeros/Less
model_9/lstm_19/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22 
model_9/lstm_19/zeros/packed/1У
model_9/lstm_19/zeros/packedPack&model_9/lstm_19/strided_slice:output:0'model_9/lstm_19/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
model_9/lstm_19/zeros/packed
model_9/lstm_19/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model_9/lstm_19/zeros/ConstЕ
model_9/lstm_19/zerosFill%model_9/lstm_19/zeros/packed:output:0$model_9/lstm_19/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
model_9/lstm_19/zeros
model_9/lstm_19/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
model_9/lstm_19/zeros_1/mul/yВ
model_9/lstm_19/zeros_1/mulMul&model_9/lstm_19/strided_slice:output:0&model_9/lstm_19/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
model_9/lstm_19/zeros_1/mul
model_9/lstm_19/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2 
model_9/lstm_19/zeros_1/Less/yЏ
model_9/lstm_19/zeros_1/LessLessmodel_9/lstm_19/zeros_1/mul:z:0'model_9/lstm_19/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
model_9/lstm_19/zeros_1/Less
 model_9/lstm_19/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22"
 model_9/lstm_19/zeros_1/packed/1Щ
model_9/lstm_19/zeros_1/packedPack&model_9/lstm_19/strided_slice:output:0)model_9/lstm_19/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
model_9/lstm_19/zeros_1/packed
model_9/lstm_19/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model_9/lstm_19/zeros_1/ConstН
model_9/lstm_19/zeros_1Fill'model_9/lstm_19/zeros_1/packed:output:0&model_9/lstm_19/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
model_9/lstm_19/zeros_1
model_9/lstm_19/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
model_9/lstm_19/transpose/permб
model_9/lstm_19/transpose	Transpose$model_9/dropout_18/Identity:output:0'model_9/lstm_19/transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
model_9/lstm_19/transpose
model_9/lstm_19/Shape_1Shapemodel_9/lstm_19/transpose:y:0*
T0*
_output_shapes
:2
model_9/lstm_19/Shape_1
%model_9/lstm_19/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%model_9/lstm_19/strided_slice_1/stack
'model_9/lstm_19/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'model_9/lstm_19/strided_slice_1/stack_1
'model_9/lstm_19/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'model_9/lstm_19/strided_slice_1/stack_2Ю
model_9/lstm_19/strided_slice_1StridedSlice model_9/lstm_19/Shape_1:output:0.model_9/lstm_19/strided_slice_1/stack:output:00model_9/lstm_19/strided_slice_1/stack_1:output:00model_9/lstm_19/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
model_9/lstm_19/strided_slice_1Ѕ
+model_9/lstm_19/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2-
+model_9/lstm_19/TensorArrayV2/element_shapeђ
model_9/lstm_19/TensorArrayV2TensorListReserve4model_9/lstm_19/TensorArrayV2/element_shape:output:0(model_9/lstm_19/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
model_9/lstm_19/TensorArrayV2п
Emodel_9/lstm_19/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   2G
Emodel_9/lstm_19/TensorArrayUnstack/TensorListFromTensor/element_shapeИ
7model_9/lstm_19/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel_9/lstm_19/transpose:y:0Nmodel_9/lstm_19/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7model_9/lstm_19/TensorArrayUnstack/TensorListFromTensor
%model_9/lstm_19/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%model_9/lstm_19/strided_slice_2/stack
'model_9/lstm_19/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'model_9/lstm_19/strided_slice_2/stack_1
'model_9/lstm_19/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'model_9/lstm_19/strided_slice_2/stack_2м
model_9/lstm_19/strided_slice_2StridedSlicemodel_9/lstm_19/transpose:y:0.model_9/lstm_19/strided_slice_2/stack:output:00model_9/lstm_19/strided_slice_2/stack_1:output:00model_9/lstm_19/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџK*
shrink_axis_mask2!
model_9/lstm_19/strided_slice_2х
2model_9/lstm_19/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp;model_9_lstm_19_lstm_cell_19_matmul_readvariableop_resource*
_output_shapes
:	KШ*
dtype024
2model_9/lstm_19/lstm_cell_19/MatMul/ReadVariableOpэ
#model_9/lstm_19/lstm_cell_19/MatMulMatMul(model_9/lstm_19/strided_slice_2:output:0:model_9/lstm_19/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2%
#model_9/lstm_19/lstm_cell_19/MatMulы
4model_9/lstm_19/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp=model_9_lstm_19_lstm_cell_19_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype026
4model_9/lstm_19/lstm_cell_19/MatMul_1/ReadVariableOpщ
%model_9/lstm_19/lstm_cell_19/MatMul_1MatMulmodel_9/lstm_19/zeros:output:0<model_9/lstm_19/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2'
%model_9/lstm_19/lstm_cell_19/MatMul_1р
 model_9/lstm_19/lstm_cell_19/addAddV2-model_9/lstm_19/lstm_cell_19/MatMul:product:0/model_9/lstm_19/lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2"
 model_9/lstm_19/lstm_cell_19/addф
3model_9/lstm_19/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp<model_9_lstm_19_lstm_cell_19_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype025
3model_9/lstm_19/lstm_cell_19/BiasAdd/ReadVariableOpэ
$model_9/lstm_19/lstm_cell_19/BiasAddBiasAdd$model_9/lstm_19/lstm_cell_19/add:z:0;model_9/lstm_19/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2&
$model_9/lstm_19/lstm_cell_19/BiasAdd
"model_9/lstm_19/lstm_cell_19/ConstConst*
_output_shapes
: *
dtype0*
value	B :2$
"model_9/lstm_19/lstm_cell_19/Const
,model_9/lstm_19/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,model_9/lstm_19/lstm_cell_19/split/split_dimГ
"model_9/lstm_19/lstm_cell_19/splitSplit5model_9/lstm_19/lstm_cell_19/split/split_dim:output:0-model_9/lstm_19/lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2$
"model_9/lstm_19/lstm_cell_19/splitЖ
$model_9/lstm_19/lstm_cell_19/SigmoidSigmoid+model_9/lstm_19/lstm_cell_19/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22&
$model_9/lstm_19/lstm_cell_19/SigmoidК
&model_9/lstm_19/lstm_cell_19/Sigmoid_1Sigmoid+model_9/lstm_19/lstm_cell_19/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22(
&model_9/lstm_19/lstm_cell_19/Sigmoid_1Ы
 model_9/lstm_19/lstm_cell_19/mulMul*model_9/lstm_19/lstm_cell_19/Sigmoid_1:y:0 model_9/lstm_19/zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22"
 model_9/lstm_19/lstm_cell_19/mul­
!model_9/lstm_19/lstm_cell_19/ReluRelu+model_9/lstm_19/lstm_cell_19/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22#
!model_9/lstm_19/lstm_cell_19/Reluм
"model_9/lstm_19/lstm_cell_19/mul_1Mul(model_9/lstm_19/lstm_cell_19/Sigmoid:y:0/model_9/lstm_19/lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"model_9/lstm_19/lstm_cell_19/mul_1б
"model_9/lstm_19/lstm_cell_19/add_1AddV2$model_9/lstm_19/lstm_cell_19/mul:z:0&model_9/lstm_19/lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"model_9/lstm_19/lstm_cell_19/add_1К
&model_9/lstm_19/lstm_cell_19/Sigmoid_2Sigmoid+model_9/lstm_19/lstm_cell_19/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22(
&model_9/lstm_19/lstm_cell_19/Sigmoid_2Ќ
#model_9/lstm_19/lstm_cell_19/Relu_1Relu&model_9/lstm_19/lstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22%
#model_9/lstm_19/lstm_cell_19/Relu_1р
"model_9/lstm_19/lstm_cell_19/mul_2Mul*model_9/lstm_19/lstm_cell_19/Sigmoid_2:y:01model_9/lstm_19/lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"model_9/lstm_19/lstm_cell_19/mul_2Џ
-model_9/lstm_19/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2/
-model_9/lstm_19/TensorArrayV2_1/element_shapeј
model_9/lstm_19/TensorArrayV2_1TensorListReserve6model_9/lstm_19/TensorArrayV2_1/element_shape:output:0(model_9/lstm_19/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
model_9/lstm_19/TensorArrayV2_1n
model_9/lstm_19/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
model_9/lstm_19/time
(model_9/lstm_19/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2*
(model_9/lstm_19/while/maximum_iterations
"model_9/lstm_19/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"model_9/lstm_19/while/loop_counterт
model_9/lstm_19/whileWhile+model_9/lstm_19/while/loop_counter:output:01model_9/lstm_19/while/maximum_iterations:output:0model_9/lstm_19/time:output:0(model_9/lstm_19/TensorArrayV2_1:handle:0model_9/lstm_19/zeros:output:0 model_9/lstm_19/zeros_1:output:0(model_9/lstm_19/strided_slice_1:output:0Gmodel_9/lstm_19/TensorArrayUnstack/TensorListFromTensor:output_handle:0;model_9_lstm_19_lstm_cell_19_matmul_readvariableop_resource=model_9_lstm_19_lstm_cell_19_matmul_1_readvariableop_resource<model_9_lstm_19_lstm_cell_19_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*/
body'R%
#model_9_lstm_19_while_body_44725412*/
cond'R%
#model_9_lstm_19_while_cond_44725411*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
model_9/lstm_19/whileе
@model_9/lstm_19/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2B
@model_9/lstm_19/TensorArrayV2Stack/TensorListStack/element_shapeБ
2model_9/lstm_19/TensorArrayV2Stack/TensorListStackTensorListStackmodel_9/lstm_19/while:output:3Imodel_9/lstm_19/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype024
2model_9/lstm_19/TensorArrayV2Stack/TensorListStackЁ
%model_9/lstm_19/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2'
%model_9/lstm_19/strided_slice_3/stack
'model_9/lstm_19/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'model_9/lstm_19/strided_slice_3/stack_1
'model_9/lstm_19/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'model_9/lstm_19/strided_slice_3/stack_2њ
model_9/lstm_19/strided_slice_3StridedSlice;model_9/lstm_19/TensorArrayV2Stack/TensorListStack:tensor:0.model_9/lstm_19/strided_slice_3/stack:output:00model_9/lstm_19/strided_slice_3/stack_1:output:00model_9/lstm_19/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2!
model_9/lstm_19/strided_slice_3
 model_9/lstm_19/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 model_9/lstm_19/transpose_1/permю
model_9/lstm_19/transpose_1	Transpose;model_9/lstm_19/TensorArrayV2Stack/TensorListStack:tensor:0)model_9/lstm_19/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
model_9/lstm_19/transpose_1
model_9/lstm_19/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
model_9/lstm_19/runtimeР
&model_9/dense_27/MatMul/ReadVariableOpReadVariableOp/model_9_dense_27_matmul_readvariableop_resource*
_output_shapes

:2 *
dtype02(
&model_9/dense_27/MatMul/ReadVariableOpШ
model_9/dense_27/MatMulMatMul(model_9/lstm_19/strided_slice_3:output:0.model_9/dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
model_9/dense_27/MatMulП
'model_9/dense_27/BiasAdd/ReadVariableOpReadVariableOp0model_9_dense_27_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'model_9/dense_27/BiasAdd/ReadVariableOpХ
model_9/dense_27/BiasAddBiasAdd!model_9/dense_27/MatMul:product:0/model_9/dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
model_9/dense_27/BiasAdd
model_9/dense_27/ReluRelu!model_9/dense_27/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
model_9/dense_27/ReluР
&model_9/dense_28/MatMul/ReadVariableOpReadVariableOp/model_9_dense_28_matmul_readvariableop_resource*
_output_shapes

:2@*
dtype02(
&model_9/dense_28/MatMul/ReadVariableOpФ
model_9/dense_28/MatMulMatMul$model_9/dropout_19/Identity:output:0.model_9/dense_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
model_9/dense_28/MatMulП
'model_9/dense_28/BiasAdd/ReadVariableOpReadVariableOp0model_9_dense_28_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model_9/dense_28/BiasAdd/ReadVariableOpХ
model_9/dense_28/BiasAddBiasAdd!model_9/dense_28/MatMul:product:0/model_9/dense_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
model_9/dense_28/BiasAdd
model_9/dense_28/ReluRelu!model_9/dense_28/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
model_9/dense_28/Relu
!model_9/concatenate_9/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_9/concatenate_9/concat/axisљ
model_9/concatenate_9/concatConcatV2#model_9/dense_27/Relu:activations:0#model_9/dense_28/Relu:activations:0*model_9/concatenate_9/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ`2
model_9/concatenate_9/concatР
&model_9/dense_29/MatMul/ReadVariableOpReadVariableOp/model_9_dense_29_matmul_readvariableop_resource*
_output_shapes

:`*
dtype02(
&model_9/dense_29/MatMul/ReadVariableOpХ
model_9/dense_29/MatMulMatMul%model_9/concatenate_9/concat:output:0.model_9/dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model_9/dense_29/MatMulП
'model_9/dense_29/BiasAdd/ReadVariableOpReadVariableOp0model_9_dense_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_9/dense_29/BiasAdd/ReadVariableOpХ
model_9/dense_29/BiasAddBiasAdd!model_9/dense_29/MatMul:product:0/model_9/dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model_9/dense_29/BiasAdd
IdentityIdentity!model_9/dense_29/BiasAdd:output:0(^model_9/dense_27/BiasAdd/ReadVariableOp'^model_9/dense_27/MatMul/ReadVariableOp(^model_9/dense_28/BiasAdd/ReadVariableOp'^model_9/dense_28/MatMul/ReadVariableOp(^model_9/dense_29/BiasAdd/ReadVariableOp'^model_9/dense_29/MatMul/ReadVariableOp/^model_9/gru_9/gru_cell_9/MatMul/ReadVariableOp1^model_9/gru_9/gru_cell_9/MatMul_1/ReadVariableOp(^model_9/gru_9/gru_cell_9/ReadVariableOp^model_9/gru_9/while4^model_9/lstm_18/lstm_cell_18/BiasAdd/ReadVariableOp3^model_9/lstm_18/lstm_cell_18/MatMul/ReadVariableOp5^model_9/lstm_18/lstm_cell_18/MatMul_1/ReadVariableOp^model_9/lstm_18/while4^model_9/lstm_19/lstm_cell_19/BiasAdd/ReadVariableOp3^model_9/lstm_19/lstm_cell_19/MatMul/ReadVariableOp5^model_9/lstm_19/lstm_cell_19/MatMul_1/ReadVariableOp^model_9/lstm_19/while*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:џџџџџџџџџџџџџџџџџџ:::::::::::::::2R
'model_9/dense_27/BiasAdd/ReadVariableOp'model_9/dense_27/BiasAdd/ReadVariableOp2P
&model_9/dense_27/MatMul/ReadVariableOp&model_9/dense_27/MatMul/ReadVariableOp2R
'model_9/dense_28/BiasAdd/ReadVariableOp'model_9/dense_28/BiasAdd/ReadVariableOp2P
&model_9/dense_28/MatMul/ReadVariableOp&model_9/dense_28/MatMul/ReadVariableOp2R
'model_9/dense_29/BiasAdd/ReadVariableOp'model_9/dense_29/BiasAdd/ReadVariableOp2P
&model_9/dense_29/MatMul/ReadVariableOp&model_9/dense_29/MatMul/ReadVariableOp2`
.model_9/gru_9/gru_cell_9/MatMul/ReadVariableOp.model_9/gru_9/gru_cell_9/MatMul/ReadVariableOp2d
0model_9/gru_9/gru_cell_9/MatMul_1/ReadVariableOp0model_9/gru_9/gru_cell_9/MatMul_1/ReadVariableOp2R
'model_9/gru_9/gru_cell_9/ReadVariableOp'model_9/gru_9/gru_cell_9/ReadVariableOp2*
model_9/gru_9/whilemodel_9/gru_9/while2j
3model_9/lstm_18/lstm_cell_18/BiasAdd/ReadVariableOp3model_9/lstm_18/lstm_cell_18/BiasAdd/ReadVariableOp2h
2model_9/lstm_18/lstm_cell_18/MatMul/ReadVariableOp2model_9/lstm_18/lstm_cell_18/MatMul/ReadVariableOp2l
4model_9/lstm_18/lstm_cell_18/MatMul_1/ReadVariableOp4model_9/lstm_18/lstm_cell_18/MatMul_1/ReadVariableOp2.
model_9/lstm_18/whilemodel_9/lstm_18/while2j
3model_9/lstm_19/lstm_cell_19/BiasAdd/ReadVariableOp3model_9/lstm_19/lstm_cell_19/BiasAdd/ReadVariableOp2h
2model_9/lstm_19/lstm_cell_19/MatMul/ReadVariableOp2model_9/lstm_19/lstm_cell_19/MatMul/ReadVariableOp2l
4model_9/lstm_19/lstm_cell_19/MatMul_1/ReadVariableOp4model_9/lstm_19/lstm_cell_19/MatMul_1/ReadVariableOp2.
model_9/lstm_19/whilemodel_9/lstm_19/while:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
input_10
C

while_body_44729991
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_18_matmul_readvariableop_resource_09
5while_lstm_cell_18_matmul_1_readvariableop_resource_08
4while_lstm_cell_18_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_18_matmul_readvariableop_resource7
3while_lstm_cell_18_matmul_1_readvariableop_resource6
2while_lstm_cell_18_biasadd_readvariableop_resourceЂ)while/lstm_cell_18/BiasAdd/ReadVariableOpЂ(while/lstm_cell_18/MatMul/ReadVariableOpЂ*while/lstm_cell_18/MatMul_1/ReadVariableOpУ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeг
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemЩ
(while/lstm_cell_18/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_18_matmul_readvariableop_resource_0*
_output_shapes
:	Ќ*
dtype02*
(while/lstm_cell_18/MatMul/ReadVariableOpз
while/lstm_cell_18/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_18/MatMulЯ
*while/lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_18_matmul_1_readvariableop_resource_0*
_output_shapes
:	KЌ*
dtype02,
*while/lstm_cell_18/MatMul_1/ReadVariableOpР
while/lstm_cell_18/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_18/MatMul_1И
while/lstm_cell_18/addAddV2#while/lstm_cell_18/MatMul:product:0%while/lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_18/addШ
)while/lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_18_biasadd_readvariableop_resource_0*
_output_shapes	
:Ќ*
dtype02+
)while/lstm_cell_18/BiasAdd/ReadVariableOpХ
while/lstm_cell_18/BiasAddBiasAddwhile/lstm_cell_18/add:z:01while/lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2
while/lstm_cell_18/BiasAddv
while/lstm_cell_18/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_18/Const
"while/lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_18/split/split_dim
while/lstm_cell_18/splitSplit+while/lstm_cell_18/split/split_dim:output:0#while/lstm_cell_18/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*
	num_split2
while/lstm_cell_18/split
while/lstm_cell_18/SigmoidSigmoid!while/lstm_cell_18/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/Sigmoid
while/lstm_cell_18/Sigmoid_1Sigmoid!while/lstm_cell_18/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/Sigmoid_1 
while/lstm_cell_18/mulMul while/lstm_cell_18/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/mul
while/lstm_cell_18/ReluRelu!while/lstm_cell_18/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/ReluД
while/lstm_cell_18/mul_1Mulwhile/lstm_cell_18/Sigmoid:y:0%while/lstm_cell_18/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/mul_1Љ
while/lstm_cell_18/add_1AddV2while/lstm_cell_18/mul:z:0while/lstm_cell_18/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/add_1
while/lstm_cell_18/Sigmoid_2Sigmoid!while/lstm_cell_18/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/Sigmoid_2
while/lstm_cell_18/Relu_1Reluwhile/lstm_cell_18/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/Relu_1И
while/lstm_cell_18/mul_2Mul while/lstm_cell_18/Sigmoid_2:y:0'while/lstm_cell_18/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2
while/lstm_cell_18/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_18/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1т
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_18/BiasAdd/ReadVariableOp)^while/lstm_cell_18/MatMul/ReadVariableOp+^while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityѕ
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_18/BiasAdd/ReadVariableOp)^while/lstm_cell_18/MatMul/ReadVariableOp+^while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1ф
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_18/BiasAdd/ReadVariableOp)^while/lstm_cell_18/MatMul/ReadVariableOp+^while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_18/BiasAdd/ReadVariableOp)^while/lstm_cell_18/MatMul/ReadVariableOp+^while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_18/mul_2:z:0*^while/lstm_cell_18/BiasAdd/ReadVariableOp)^while/lstm_cell_18/MatMul/ReadVariableOp+^while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_18/add_1:z:0*^while/lstm_cell_18/BiasAdd/ReadVariableOp)^while/lstm_cell_18/MatMul/ReadVariableOp+^while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_18_biasadd_readvariableop_resource4while_lstm_cell_18_biasadd_readvariableop_resource_0"l
3while_lstm_cell_18_matmul_1_readvariableop_resource5while_lstm_cell_18_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_18_matmul_readvariableop_resource3while_lstm_cell_18_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџK:џџџџџџџџџK: : :::2V
)while/lstm_cell_18/BiasAdd/ReadVariableOp)while/lstm_cell_18/BiasAdd/ReadVariableOp2T
(while/lstm_cell_18/MatMul/ReadVariableOp(while/lstm_cell_18/MatMul/ReadVariableOp2X
*while/lstm_cell_18/MatMul_1/ReadVariableOp*while/lstm_cell_18/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџK:-)
'
_output_shapes
:џџџџџџџџџK:

_output_shapes
: :

_output_shapes
: 
Е
Э
while_cond_44727524
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_44727524___redundant_placeholder06
2while_while_cond_44727524___redundant_placeholder16
2while_while_cond_44727524___redundant_placeholder26
2while_while_cond_44727524___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџK:џџџџџџџџџK: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџK:-)
'
_output_shapes
:џџџџџџџџџK:

_output_shapes
: :

_output_shapes
:
јD
ф
E__inference_lstm_19_layer_call_and_return_conditional_losses_44727292

inputs
lstm_cell_19_44727210
lstm_cell_19_44727212
lstm_cell_19_44727214
identityЂ$lstm_cell_19/StatefulPartitionedCallЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџK*
shrink_axis_mask2
strided_slice_2Ј
$lstm_cell_19/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_19_44727210lstm_cell_19_44727212lstm_cell_19_44727214*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_lstm_cell_19_layer_call_and_return_conditional_losses_447267972&
$lstm_cell_19/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter­
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_19_44727210lstm_cell_19_44727212lstm_cell_19_44727214*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_44727223*
condR
while_cond_44727222*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime
IdentityIdentitystrided_slice_3:output:0%^lstm_cell_19/StatefulPartitionedCall^while*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџK:::2L
$lstm_cell_19/StatefulPartitionedCall$lstm_cell_19/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK
 
_user_specified_nameinputs

g
H__inference_dropout_19_layer_call_and_return_conditional_losses_44731801

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeД
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ22
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ2:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
щZ
ж
C__inference_gru_9_layer_call_and_return_conditional_losses_44727798

inputs&
"gru_cell_9_readvariableop_resource-
)gru_cell_9_matmul_readvariableop_resource/
+gru_cell_9_matmul_1_readvariableop_resource
identityЂ gru_cell_9/MatMul/ReadVariableOpЂ"gru_cell_9/MatMul_1/ReadVariableOpЂgru_cell_9/ReadVariableOpЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2
gru_cell_9/ReadVariableOpReadVariableOp"gru_cell_9_readvariableop_resource*
_output_shapes
:	*
dtype02
gru_cell_9/ReadVariableOp
gru_cell_9/unstackUnpack!gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
gru_cell_9/unstackЏ
 gru_cell_9/MatMul/ReadVariableOpReadVariableOp)gru_cell_9_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 gru_cell_9/MatMul/ReadVariableOpЇ
gru_cell_9/MatMulMatMulstrided_slice_2:output:0(gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_cell_9/MatMul 
gru_cell_9/BiasAddBiasAddgru_cell_9/MatMul:product:0gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_cell_9/BiasAddf
gru_cell_9/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_9/Const
gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_9/split/split_dimи
gru_cell_9/splitSplit#gru_cell_9/split/split_dim:output:0gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
gru_cell_9/splitЕ
"gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_9_matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype02$
"gru_cell_9/MatMul_1/ReadVariableOpЃ
gru_cell_9/MatMul_1MatMulzeros:output:0*gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_cell_9/MatMul_1І
gru_cell_9/BiasAdd_1BiasAddgru_cell_9/MatMul_1:product:0gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:џџџџџџџџџ2
gru_cell_9/BiasAdd_1}
gru_cell_9/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   џџџџ2
gru_cell_9/Const_1
gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_9/split_1/split_dim
gru_cell_9/split_1SplitVgru_cell_9/BiasAdd_1:output:0gru_cell_9/Const_1:output:0%gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
gru_cell_9/split_1
gru_cell_9/addAddV2gru_cell_9/split:output:0gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/addy
gru_cell_9/SigmoidSigmoidgru_cell_9/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/Sigmoid
gru_cell_9/add_1AddV2gru_cell_9/split:output:1gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/add_1
gru_cell_9/Sigmoid_1Sigmoidgru_cell_9/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/Sigmoid_1
gru_cell_9/mulMulgru_cell_9/Sigmoid_1:y:0gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/mul
gru_cell_9/add_2AddV2gru_cell_9/split:output:2gru_cell_9/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/add_2r
gru_cell_9/ReluRelugru_cell_9/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/Relu
gru_cell_9/mul_1Mulgru_cell_9/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/mul_1i
gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_9/sub/x
gru_cell_9/subSubgru_cell_9/sub/x:output:0gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/sub
gru_cell_9/mul_2Mulgru_cell_9/sub:z:0gru_cell_9/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/mul_2
gru_cell_9/add_3AddV2gru_cell_9/mul_1:z:0gru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
gru_cell_9/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЋ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_9_readvariableop_resource)gru_cell_9_matmul_readvariableop_resource+gru_cell_9_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_44727708*
condR
while_cond_44727707*8
output_shapes'
%: : : : :џџџџџџџџџ2: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeи
IdentityIdentitystrided_slice_3:output:0!^gru_cell_9/MatMul/ReadVariableOp#^gru_cell_9/MatMul_1/ReadVariableOp^gru_cell_9/ReadVariableOp^while*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2D
 gru_cell_9/MatMul/ReadVariableOp gru_cell_9/MatMul/ReadVariableOp2H
"gru_cell_9/MatMul_1/ReadVariableOp"gru_cell_9/MatMul_1/ReadVariableOp26
gru_cell_9/ReadVariableOpgru_cell_9/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ђ

О
!model_9_gru_9_while_cond_447252558
4model_9_gru_9_while_model_9_gru_9_while_loop_counter>
:model_9_gru_9_while_model_9_gru_9_while_maximum_iterations#
model_9_gru_9_while_placeholder%
!model_9_gru_9_while_placeholder_1%
!model_9_gru_9_while_placeholder_2:
6model_9_gru_9_while_less_model_9_gru_9_strided_slice_1R
Nmodel_9_gru_9_while_model_9_gru_9_while_cond_44725255___redundant_placeholder0R
Nmodel_9_gru_9_while_model_9_gru_9_while_cond_44725255___redundant_placeholder1R
Nmodel_9_gru_9_while_model_9_gru_9_while_cond_44725255___redundant_placeholder2R
Nmodel_9_gru_9_while_model_9_gru_9_while_cond_44725255___redundant_placeholder3 
model_9_gru_9_while_identity
Ж
model_9/gru_9/while/LessLessmodel_9_gru_9_while_placeholder6model_9_gru_9_while_less_model_9_gru_9_strided_slice_1*
T0*
_output_shapes
: 2
model_9/gru_9/while/Less
model_9/gru_9/while/IdentityIdentitymodel_9/gru_9/while/Less:z:0*
T0
*
_output_shapes
: 2
model_9/gru_9/while/Identity"E
model_9_gru_9_while_identity%model_9/gru_9/while/Identity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
O


lstm_18_while_body_44729287,
(lstm_18_while_lstm_18_while_loop_counter2
.lstm_18_while_lstm_18_while_maximum_iterations
lstm_18_while_placeholder
lstm_18_while_placeholder_1
lstm_18_while_placeholder_2
lstm_18_while_placeholder_3+
'lstm_18_while_lstm_18_strided_slice_1_0g
clstm_18_while_tensorarrayv2read_tensorlistgetitem_lstm_18_tensorarrayunstack_tensorlistfromtensor_0?
;lstm_18_while_lstm_cell_18_matmul_readvariableop_resource_0A
=lstm_18_while_lstm_cell_18_matmul_1_readvariableop_resource_0@
<lstm_18_while_lstm_cell_18_biasadd_readvariableop_resource_0
lstm_18_while_identity
lstm_18_while_identity_1
lstm_18_while_identity_2
lstm_18_while_identity_3
lstm_18_while_identity_4
lstm_18_while_identity_5)
%lstm_18_while_lstm_18_strided_slice_1e
alstm_18_while_tensorarrayv2read_tensorlistgetitem_lstm_18_tensorarrayunstack_tensorlistfromtensor=
9lstm_18_while_lstm_cell_18_matmul_readvariableop_resource?
;lstm_18_while_lstm_cell_18_matmul_1_readvariableop_resource>
:lstm_18_while_lstm_cell_18_biasadd_readvariableop_resourceЂ1lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOpЂ0lstm_18/while/lstm_cell_18/MatMul/ReadVariableOpЂ2lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOpг
?lstm_18/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2A
?lstm_18/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_18/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_18_while_tensorarrayv2read_tensorlistgetitem_lstm_18_tensorarrayunstack_tensorlistfromtensor_0lstm_18_while_placeholderHlstm_18/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype023
1lstm_18/while/TensorArrayV2Read/TensorListGetItemс
0lstm_18/while/lstm_cell_18/MatMul/ReadVariableOpReadVariableOp;lstm_18_while_lstm_cell_18_matmul_readvariableop_resource_0*
_output_shapes
:	Ќ*
dtype022
0lstm_18/while/lstm_cell_18/MatMul/ReadVariableOpї
!lstm_18/while/lstm_cell_18/MatMulMatMul8lstm_18/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2#
!lstm_18/while/lstm_cell_18/MatMulч
2lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp=lstm_18_while_lstm_cell_18_matmul_1_readvariableop_resource_0*
_output_shapes
:	KЌ*
dtype024
2lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOpр
#lstm_18/while/lstm_cell_18/MatMul_1MatMullstm_18_while_placeholder_2:lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2%
#lstm_18/while/lstm_cell_18/MatMul_1и
lstm_18/while/lstm_cell_18/addAddV2+lstm_18/while/lstm_cell_18/MatMul:product:0-lstm_18/while/lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2 
lstm_18/while/lstm_cell_18/addр
1lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp<lstm_18_while_lstm_cell_18_biasadd_readvariableop_resource_0*
_output_shapes	
:Ќ*
dtype023
1lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOpх
"lstm_18/while/lstm_cell_18/BiasAddBiasAdd"lstm_18/while/lstm_cell_18/add:z:09lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЌ2$
"lstm_18/while/lstm_cell_18/BiasAdd
 lstm_18/while/lstm_cell_18/ConstConst*
_output_shapes
: *
dtype0*
value	B :2"
 lstm_18/while/lstm_cell_18/Const
*lstm_18/while/lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_18/while/lstm_cell_18/split/split_dimЋ
 lstm_18/while/lstm_cell_18/splitSplit3lstm_18/while/lstm_cell_18/split/split_dim:output:0+lstm_18/while/lstm_cell_18/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK:џџџџџџџџџK*
	num_split2"
 lstm_18/while/lstm_cell_18/splitА
"lstm_18/while/lstm_cell_18/SigmoidSigmoid)lstm_18/while/lstm_cell_18/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџK2$
"lstm_18/while/lstm_cell_18/SigmoidД
$lstm_18/while/lstm_cell_18/Sigmoid_1Sigmoid)lstm_18/while/lstm_cell_18/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџK2&
$lstm_18/while/lstm_cell_18/Sigmoid_1Р
lstm_18/while/lstm_cell_18/mulMul(lstm_18/while/lstm_cell_18/Sigmoid_1:y:0lstm_18_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџK2 
lstm_18/while/lstm_cell_18/mulЇ
lstm_18/while/lstm_cell_18/ReluRelu)lstm_18/while/lstm_cell_18/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџK2!
lstm_18/while/lstm_cell_18/Reluд
 lstm_18/while/lstm_cell_18/mul_1Mul&lstm_18/while/lstm_cell_18/Sigmoid:y:0-lstm_18/while/lstm_cell_18/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2"
 lstm_18/while/lstm_cell_18/mul_1Щ
 lstm_18/while/lstm_cell_18/add_1AddV2"lstm_18/while/lstm_cell_18/mul:z:0$lstm_18/while/lstm_cell_18/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2"
 lstm_18/while/lstm_cell_18/add_1Д
$lstm_18/while/lstm_cell_18/Sigmoid_2Sigmoid)lstm_18/while/lstm_cell_18/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџK2&
$lstm_18/while/lstm_cell_18/Sigmoid_2І
!lstm_18/while/lstm_cell_18/Relu_1Relu$lstm_18/while/lstm_cell_18/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџK2#
!lstm_18/while/lstm_cell_18/Relu_1и
 lstm_18/while/lstm_cell_18/mul_2Mul(lstm_18/while/lstm_cell_18/Sigmoid_2:y:0/lstm_18/while/lstm_cell_18/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџK2"
 lstm_18/while/lstm_cell_18/mul_2
2lstm_18/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_18_while_placeholder_1lstm_18_while_placeholder$lstm_18/while/lstm_cell_18/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_18/while/TensorArrayV2Write/TensorListSetIteml
lstm_18/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_18/while/add/y
lstm_18/while/addAddV2lstm_18_while_placeholderlstm_18/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_18/while/addp
lstm_18/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_18/while/add_1/y
lstm_18/while/add_1AddV2(lstm_18_while_lstm_18_while_loop_counterlstm_18/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_18/while/add_1
lstm_18/while/IdentityIdentitylstm_18/while/add_1:z:02^lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOp1^lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp3^lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_18/while/Identity­
lstm_18/while/Identity_1Identity.lstm_18_while_lstm_18_while_maximum_iterations2^lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOp1^lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp3^lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_18/while/Identity_1
lstm_18/while/Identity_2Identitylstm_18/while/add:z:02^lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOp1^lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp3^lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_18/while/Identity_2С
lstm_18/while/Identity_3IdentityBlstm_18/while/TensorArrayV2Write/TensorListSetItem:output_handle:02^lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOp1^lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp3^lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm_18/while/Identity_3Д
lstm_18/while/Identity_4Identity$lstm_18/while/lstm_cell_18/mul_2:z:02^lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOp1^lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp3^lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_18/while/Identity_4Д
lstm_18/while/Identity_5Identity$lstm_18/while/lstm_cell_18/add_1:z:02^lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOp1^lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp3^lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџK2
lstm_18/while/Identity_5"9
lstm_18_while_identitylstm_18/while/Identity:output:0"=
lstm_18_while_identity_1!lstm_18/while/Identity_1:output:0"=
lstm_18_while_identity_2!lstm_18/while/Identity_2:output:0"=
lstm_18_while_identity_3!lstm_18/while/Identity_3:output:0"=
lstm_18_while_identity_4!lstm_18/while/Identity_4:output:0"=
lstm_18_while_identity_5!lstm_18/while/Identity_5:output:0"P
%lstm_18_while_lstm_18_strided_slice_1'lstm_18_while_lstm_18_strided_slice_1_0"z
:lstm_18_while_lstm_cell_18_biasadd_readvariableop_resource<lstm_18_while_lstm_cell_18_biasadd_readvariableop_resource_0"|
;lstm_18_while_lstm_cell_18_matmul_1_readvariableop_resource=lstm_18_while_lstm_cell_18_matmul_1_readvariableop_resource_0"x
9lstm_18_while_lstm_cell_18_matmul_readvariableop_resource;lstm_18_while_lstm_cell_18_matmul_readvariableop_resource_0"Ш
alstm_18_while_tensorarrayv2read_tensorlistgetitem_lstm_18_tensorarrayunstack_tensorlistfromtensorclstm_18_while_tensorarrayv2read_tensorlistgetitem_lstm_18_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :џџџџџџџџџK:џџџџџџџџџK: : :::2f
1lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOp1lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOp2d
0lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp0lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp2h
2lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOp2lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџK:-)
'
_output_shapes
:џџџџџџџџџK:

_output_shapes
: :

_output_shapes
: 


(__inference_gru_9_layer_call_fn_44731133
inputs_0
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_gru_9_layer_call_and_return_conditional_losses_447266822
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
І

э
lstm_18_while_cond_44729286,
(lstm_18_while_lstm_18_while_loop_counter2
.lstm_18_while_lstm_18_while_maximum_iterations
lstm_18_while_placeholder
lstm_18_while_placeholder_1
lstm_18_while_placeholder_2
lstm_18_while_placeholder_3.
*lstm_18_while_less_lstm_18_strided_slice_1F
Blstm_18_while_lstm_18_while_cond_44729286___redundant_placeholder0F
Blstm_18_while_lstm_18_while_cond_44729286___redundant_placeholder1F
Blstm_18_while_lstm_18_while_cond_44729286___redundant_placeholder2F
Blstm_18_while_lstm_18_while_cond_44729286___redundant_placeholder3
lstm_18_while_identity

lstm_18/while/LessLesslstm_18_while_placeholder*lstm_18_while_less_lstm_18_strided_slice_1*
T0*
_output_shapes
: 2
lstm_18/while/Lessu
lstm_18/while/IdentityIdentitylstm_18/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_18/while/Identity"9
lstm_18_while_identitylstm_18/while/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџK:џџџџџџџџџK: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџK:-)
'
_output_shapes
:џџџџџџџџџK:

_output_shapes
: :

_output_shapes
:
у	
Џ
-__inference_gru_cell_9_layer_call_fn_44732082

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1ЂStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ2:џџџџџџџџџ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_gru_cell_9_layer_call_and_return_conditional_losses_447262012
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ2:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ2
"
_user_specified_name
states/0
Х[
є
E__inference_lstm_19_layer_call_and_return_conditional_losses_44731767

inputs/
+lstm_cell_19_matmul_readvariableop_resource1
-lstm_cell_19_matmul_1_readvariableop_resource0
,lstm_cell_19_biasadd_readvariableop_resource
identityЂ#lstm_cell_19/BiasAdd/ReadVariableOpЂ"lstm_cell_19/MatMul/ReadVariableOpЂ$lstm_cell_19/MatMul_1/ReadVariableOpЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџK*
shrink_axis_mask2
strided_slice_2Е
"lstm_cell_19/MatMul/ReadVariableOpReadVariableOp+lstm_cell_19_matmul_readvariableop_resource*
_output_shapes
:	KШ*
dtype02$
"lstm_cell_19/MatMul/ReadVariableOp­
lstm_cell_19/MatMulMatMulstrided_slice_2:output:0*lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_19/MatMulЛ
$lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_19_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02&
$lstm_cell_19/MatMul_1/ReadVariableOpЉ
lstm_cell_19/MatMul_1MatMulzeros:output:0,lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_19/MatMul_1 
lstm_cell_19/addAddV2lstm_cell_19/MatMul:product:0lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_19/addД
#lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_19_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02%
#lstm_cell_19/BiasAdd/ReadVariableOp­
lstm_cell_19/BiasAddBiasAddlstm_cell_19/add:z:0+lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_19/BiasAddj
lstm_cell_19/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_19/Const~
lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_19/split/split_dimѓ
lstm_cell_19/splitSplit%lstm_cell_19/split/split_dim:output:0lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_19/split
lstm_cell_19/SigmoidSigmoidlstm_cell_19/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/Sigmoid
lstm_cell_19/Sigmoid_1Sigmoidlstm_cell_19/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/Sigmoid_1
lstm_cell_19/mulMullstm_cell_19/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/mul}
lstm_cell_19/ReluRelulstm_cell_19/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/Relu
lstm_cell_19/mul_1Mullstm_cell_19/Sigmoid:y:0lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/mul_1
lstm_cell_19/add_1AddV2lstm_cell_19/mul:z:0lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/add_1
lstm_cell_19/Sigmoid_2Sigmoidlstm_cell_19/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/Sigmoid_2|
lstm_cell_19/Relu_1Relulstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/Relu_1 
lstm_cell_19/mul_2Mullstm_cell_19/Sigmoid_2:y:0!lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterђ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_19_matmul_readvariableop_resource-lstm_cell_19_matmul_1_readvariableop_resource,lstm_cell_19_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_44731682*
condR
while_cond_44731681*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeц
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_19/BiasAdd/ReadVariableOp#^lstm_cell_19/MatMul/ReadVariableOp%^lstm_cell_19/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџK:::2J
#lstm_cell_19/BiasAdd/ReadVariableOp#lstm_cell_19/BiasAdd/ReadVariableOp2H
"lstm_cell_19/MatMul/ReadVariableOp"lstm_cell_19/MatMul/ReadVariableOp2L
$lstm_cell_19/MatMul_1/ReadVariableOp$lstm_cell_19/MatMul_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK
 
_user_specified_nameinputs
к
Д
while_cond_44730861
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_44730861___redundant_placeholder06
2while_while_cond_44730861___redundant_placeholder16
2while_while_cond_44730861___redundant_placeholder26
2while_while_cond_44730861___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
І

э
lstm_19_while_cond_44729592,
(lstm_19_while_lstm_19_while_loop_counter2
.lstm_19_while_lstm_19_while_maximum_iterations
lstm_19_while_placeholder
lstm_19_while_placeholder_1
lstm_19_while_placeholder_2
lstm_19_while_placeholder_3.
*lstm_19_while_less_lstm_19_strided_slice_1F
Blstm_19_while_lstm_19_while_cond_44729592___redundant_placeholder0F
Blstm_19_while_lstm_19_while_cond_44729592___redundant_placeholder1F
Blstm_19_while_lstm_19_while_cond_44729592___redundant_placeholder2F
Blstm_19_while_lstm_19_while_cond_44729592___redundant_placeholder3
lstm_19_while_identity

lstm_19/while/LessLesslstm_19_while_placeholder*lstm_19_while_less_lstm_19_strided_slice_1*
T0*
_output_shapes
: 2
lstm_19/while/Lessu
lstm_19/while/IdentityIdentitylstm_19/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_19/while/Identity"9
lstm_19_while_identitylstm_19/while/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџ2:џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:
Э[
і
E__inference_lstm_19_layer_call_and_return_conditional_losses_44731286
inputs_0/
+lstm_cell_19_matmul_readvariableop_resource1
-lstm_cell_19_matmul_1_readvariableop_resource0
,lstm_cell_19_biasadd_readvariableop_resource
identityЂ#lstm_cell_19/BiasAdd/ReadVariableOpЂ"lstm_cell_19/MatMul/ReadVariableOpЂ$lstm_cell_19/MatMul_1/ReadVariableOpЂwhileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџK   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџK*
shrink_axis_mask2
strided_slice_2Е
"lstm_cell_19/MatMul/ReadVariableOpReadVariableOp+lstm_cell_19_matmul_readvariableop_resource*
_output_shapes
:	KШ*
dtype02$
"lstm_cell_19/MatMul/ReadVariableOp­
lstm_cell_19/MatMulMatMulstrided_slice_2:output:0*lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_19/MatMulЛ
$lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_19_matmul_1_readvariableop_resource*
_output_shapes
:	2Ш*
dtype02&
$lstm_cell_19/MatMul_1/ReadVariableOpЉ
lstm_cell_19/MatMul_1MatMulzeros:output:0,lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_19/MatMul_1 
lstm_cell_19/addAddV2lstm_cell_19/MatMul:product:0lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_19/addД
#lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_19_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02%
#lstm_cell_19/BiasAdd/ReadVariableOp­
lstm_cell_19/BiasAddBiasAddlstm_cell_19/add:z:0+lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
lstm_cell_19/BiasAddj
lstm_cell_19/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_19/Const~
lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_19/split/split_dimѓ
lstm_cell_19/splitSplit%lstm_cell_19/split/split_dim:output:0lstm_cell_19/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2:џџџџџџџџџ2*
	num_split2
lstm_cell_19/split
lstm_cell_19/SigmoidSigmoidlstm_cell_19/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/Sigmoid
lstm_cell_19/Sigmoid_1Sigmoidlstm_cell_19/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/Sigmoid_1
lstm_cell_19/mulMullstm_cell_19/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/mul}
lstm_cell_19/ReluRelulstm_cell_19/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/Relu
lstm_cell_19/mul_1Mullstm_cell_19/Sigmoid:y:0lstm_cell_19/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/mul_1
lstm_cell_19/add_1AddV2lstm_cell_19/mul:z:0lstm_cell_19/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/add_1
lstm_cell_19/Sigmoid_2Sigmoidlstm_cell_19/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/Sigmoid_2|
lstm_cell_19/Relu_1Relulstm_cell_19/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/Relu_1 
lstm_cell_19/mul_2Mullstm_cell_19/Sigmoid_2:y:0!lstm_cell_19/Relu_1:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
lstm_cell_19/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterђ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_19_matmul_readvariableop_resource-lstm_cell_19_matmul_1_readvariableop_resource,lstm_cell_19_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_44731201*
condR
while_cond_44731200*K
output_shapes:
8: : : : :џџџџџџџџџ2:џџџџџџџџџ2: : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ2   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeц
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_19/BiasAdd/ReadVariableOp#^lstm_cell_19/MatMul/ReadVariableOp%^lstm_cell_19/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџK:::2J
#lstm_cell_19/BiasAdd/ReadVariableOp#lstm_cell_19/BiasAdd/ReadVariableOp2H
"lstm_cell_19/MatMul/ReadVariableOp"lstm_cell_19/MatMul/ReadVariableOp2L
$lstm_cell_19/MatMul_1/ReadVariableOp$lstm_cell_19/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџK
"
_user_specified_name
inputs/0
Е
Э
while_cond_44731200
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_44731200___redundant_placeholder06
2while_while_cond_44731200___redundant_placeholder16
2while_while_cond_44731200___redundant_placeholder26
2while_while_cond_44731200___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :џџџџџџџџџ2:џџџџџџџџџ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ2:-)
'
_output_shapes
:џџџџџџџџџ2:

_output_shapes
: :

_output_shapes
:"БL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*К
serving_defaultІ
J
input_10>
serving_default_input_10:0џџџџџџџџџџџџџџџџџџ<
dense_290
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:ЁЦ
У]
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
+Ю&call_and_return_all_conditional_losses
Я_default_save_signature
а__call__"зY
_tf_keras_networkЛY{"class_name": "Functional", "name": "model_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_10"}, "name": "input_10", "inbound_nodes": []}, {"class_name": "LSTM", "config": {"name": "lstm_18", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 75, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm_18", "inbound_nodes": [[["input_10", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_18", "trainable": true, "dtype": "float32", "rate": 0.05, "noise_shape": null, "seed": null}, "name": "dropout_18", "inbound_nodes": [[["lstm_18", 0, 0, {}]]]}, {"class_name": "GRU", "config": {"name": "gru_9", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}, "name": "gru_9", "inbound_nodes": [[["input_10", 0, 0, {}]]]}, {"class_name": "LSTM", "config": {"name": "lstm_19", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm_19", "inbound_nodes": [[["dropout_18", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_19", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "name": "dropout_19", "inbound_nodes": [[["gru_9", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_27", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_27", "inbound_nodes": [[["lstm_19", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_28", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_28", "inbound_nodes": [[["dropout_19", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_9", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_9", "inbound_nodes": [[["dense_27", 0, 0, {}], ["dense_28", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_29", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_29", "inbound_nodes": [[["concatenate_9", 0, 0, {}]]]}], "input_layers": [["input_10", 0, 0]], "output_layers": [["dense_29", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_10"}, "name": "input_10", "inbound_nodes": []}, {"class_name": "LSTM", "config": {"name": "lstm_18", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 75, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm_18", "inbound_nodes": [[["input_10", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_18", "trainable": true, "dtype": "float32", "rate": 0.05, "noise_shape": null, "seed": null}, "name": "dropout_18", "inbound_nodes": [[["lstm_18", 0, 0, {}]]]}, {"class_name": "GRU", "config": {"name": "gru_9", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}, "name": "gru_9", "inbound_nodes": [[["input_10", 0, 0, {}]]]}, {"class_name": "LSTM", "config": {"name": "lstm_19", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm_19", "inbound_nodes": [[["dropout_18", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_19", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "name": "dropout_19", "inbound_nodes": [[["gru_9", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_27", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_27", "inbound_nodes": [[["lstm_19", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_28", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_28", "inbound_nodes": [[["dropout_19", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_9", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_9", "inbound_nodes": [[["dense_27", 0, 0, {}], ["dense_28", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_29", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_29", "inbound_nodes": [[["concatenate_9", 0, 0, {}]]]}], "input_layers": [["input_10", 0, 0]], "output_layers": [["dense_29", 0, 0]]}}, "training_config": {"loss": "mean_squared_error", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}}, {"class_name": "MeanMetricWrapper", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 9.999999747378752e-05, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ї"є
_tf_keras_input_layerд{"class_name": "InputLayer", "name": "input_10", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_10"}}
Р
cell

state_spec
trainable_variables
regularization_losses
	variables
	keras_api
+б&call_and_return_all_conditional_losses
в__call__"

_tf_keras_rnn_layerї	{"class_name": "LSTM", "name": "lstm_18", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_18", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 75, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 1]}}
ъ
trainable_variables
regularization_losses
	variables
	keras_api
+г&call_and_return_all_conditional_losses
д__call__"й
_tf_keras_layerП{"class_name": "Dropout", "name": "dropout_18", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_18", "trainable": true, "dtype": "float32", "rate": 0.05, "noise_shape": null, "seed": null}}
З
cell

state_spec
trainable_variables
regularization_losses
	variables
 	keras_api
+е&call_and_return_all_conditional_losses
ж__call__"

_tf_keras_rnn_layerю	{"class_name": "GRU", "name": "gru_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gru_9", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 1]}}
У
!cell
"
state_spec
#trainable_variables
$regularization_losses
%	variables
&	keras_api
+з&call_and_return_all_conditional_losses
и__call__"

_tf_keras_rnn_layerњ	{"class_name": "LSTM", "name": "lstm_19", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_19", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 75]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 75]}}
щ
'trainable_variables
(regularization_losses
)	variables
*	keras_api
+й&call_and_return_all_conditional_losses
к__call__"и
_tf_keras_layerО{"class_name": "Dropout", "name": "dropout_19", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_19", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}
є

+kernel
,bias
-trainable_variables
.regularization_losses
/	variables
0	keras_api
+л&call_and_return_all_conditional_losses
м__call__"Э
_tf_keras_layerГ{"class_name": "Dense", "name": "dense_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_27", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
є

1kernel
2bias
3trainable_variables
4regularization_losses
5	variables
6	keras_api
+н&call_and_return_all_conditional_losses
о__call__"Э
_tf_keras_layerГ{"class_name": "Dense", "name": "dense_28", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_28", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
Я
7trainable_variables
8regularization_losses
9	variables
:	keras_api
+п&call_and_return_all_conditional_losses
р__call__"О
_tf_keras_layerЄ{"class_name": "Concatenate", "name": "concatenate_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_9", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 32]}, {"class_name": "TensorShape", "items": [null, 64]}]}
ѕ

;kernel
<bias
=trainable_variables
>regularization_losses
?	variables
@	keras_api
+с&call_and_return_all_conditional_losses
т__call__"Ю
_tf_keras_layerД{"class_name": "Dense", "name": "dense_29", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_29", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 96}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 96]}}
џ
Aiter

Bbeta_1

Cbeta_2
	Ddecay
Elearning_rate+mА,mБ1mВ2mГ;mД<mЕFmЖGmЗHmИImЙJmКKmЛLmМMmНNmО+vП,vР1vС2vТ;vУ<vФFvХGvЦHvЧIvШJvЩKvЪLvЫMvЬNvЭ"
	optimizer

F0
G1
H2
I3
J4
K5
L6
M7
N8
+9
,10
111
212
;13
<14"
trackable_list_wrapper
 "
trackable_list_wrapper

F0
G1
H2
I3
J4
K5
L6
M7
N8
+9
,10
111
212
;13
<14"
trackable_list_wrapper
Ю
Olayer_metrics

Players
Qlayer_regularization_losses
trainable_variables
regularization_losses
Rnon_trainable_variables
Smetrics
	variables
а__call__
Я_default_save_signature
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses"
_generic_user_object
-
уserving_default"
signature_map
Ќ

Fkernel
Grecurrent_kernel
Hbias
Ttrainable_variables
Uregularization_losses
V	variables
W	keras_api
+ф&call_and_return_all_conditional_losses
х__call__"я
_tf_keras_layerе{"class_name": "LSTMCell", "name": "lstm_cell_18", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_18", "trainable": true, "dtype": "float32", "units": 75, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
5
F0
G1
H2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
F0
G1
H2"
trackable_list_wrapper
М
Xlayer_metrics

Ylayers
Zlayer_regularization_losses
trainable_variables
regularization_losses
[non_trainable_variables
\metrics

]states
	variables
в__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
^layer_metrics

_layers
`layer_regularization_losses
trainable_variables
regularization_losses
anon_trainable_variables
bmetrics
	variables
д__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses"
_generic_user_object
Ђ

Ikernel
Jrecurrent_kernel
Kbias
ctrainable_variables
dregularization_losses
e	variables
f	keras_api
+ц&call_and_return_all_conditional_losses
ч__call__"х
_tf_keras_layerЫ{"class_name": "GRUCell", "name": "gru_cell_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gru_cell_9", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}
 "
trackable_list_wrapper
5
I0
J1
K2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
I0
J1
K2"
trackable_list_wrapper
М
glayer_metrics

hlayers
ilayer_regularization_losses
trainable_variables
regularization_losses
jnon_trainable_variables
kmetrics

lstates
	variables
ж__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses"
_generic_user_object
Ќ

Lkernel
Mrecurrent_kernel
Nbias
mtrainable_variables
nregularization_losses
o	variables
p	keras_api
+ш&call_and_return_all_conditional_losses
щ__call__"я
_tf_keras_layerе{"class_name": "LSTMCell", "name": "lstm_cell_19", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_19", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
5
L0
M1
N2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
L0
M1
N2"
trackable_list_wrapper
М
qlayer_metrics

rlayers
slayer_regularization_losses
#trainable_variables
$regularization_losses
tnon_trainable_variables
umetrics

vstates
%	variables
и__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
wlayer_metrics

xlayers
ylayer_regularization_losses
'trainable_variables
(regularization_losses
znon_trainable_variables
{metrics
)	variables
к__call__
+й&call_and_return_all_conditional_losses
'й"call_and_return_conditional_losses"
_generic_user_object
!:2 2dense_27/kernel
: 2dense_27/bias
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
Б
|layer_metrics

}layers
~layer_regularization_losses
-trainable_variables
.regularization_losses
non_trainable_variables
metrics
/	variables
м__call__
+л&call_and_return_all_conditional_losses
'л"call_and_return_conditional_losses"
_generic_user_object
!:2@2dense_28/kernel
:@2dense_28/bias
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
Е
layer_metrics
layers
 layer_regularization_losses
3trainable_variables
4regularization_losses
non_trainable_variables
metrics
5	variables
о__call__
+н&call_and_return_all_conditional_losses
'н"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
layer_metrics
layers
 layer_regularization_losses
7trainable_variables
8regularization_losses
non_trainable_variables
metrics
9	variables
р__call__
+п&call_and_return_all_conditional_losses
'п"call_and_return_conditional_losses"
_generic_user_object
!:`2dense_29/kernel
:2dense_29/bias
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
Е
layer_metrics
layers
 layer_regularization_losses
=trainable_variables
>regularization_losses
non_trainable_variables
metrics
?	variables
т__call__
+с&call_and_return_all_conditional_losses
'с"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
.:,	Ќ2lstm_18/lstm_cell_18/kernel
8:6	KЌ2%lstm_18/lstm_cell_18/recurrent_kernel
(:&Ќ2lstm_18/lstm_cell_18/bias
*:(	2gru_9/gru_cell_9/kernel
4:2	22!gru_9/gru_cell_9/recurrent_kernel
(:&	2gru_9/gru_cell_9/bias
.:,	KШ2lstm_19/lstm_cell_19/kernel
8:6	2Ш2%lstm_19/lstm_cell_19/recurrent_kernel
(:&Ш2lstm_19/lstm_cell_19/bias
 "
trackable_dict_wrapper
f
0
1
2
3
4
5
6
7
	8

9"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
8
0
1
2"
trackable_list_wrapper
5
F0
G1
H2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
F0
G1
H2"
trackable_list_wrapper
Е
layer_metrics
layers
 layer_regularization_losses
Ttrainable_variables
Uregularization_losses
non_trainable_variables
metrics
V	variables
х__call__
+ф&call_and_return_all_conditional_losses
'ф"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
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
5
I0
J1
K2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
I0
J1
K2"
trackable_list_wrapper
Е
layer_metrics
layers
 layer_regularization_losses
ctrainable_variables
dregularization_losses
non_trainable_variables
metrics
e	variables
ч__call__
+ц&call_and_return_all_conditional_losses
'ц"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
L0
M1
N2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
L0
M1
N2"
trackable_list_wrapper
Е
layer_metrics
layers
 layer_regularization_losses
mtrainable_variables
nregularization_losses
 non_trainable_variables
Ёmetrics
o	variables
щ__call__
+ш&call_and_return_all_conditional_losses
'ш"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
'
!0"
trackable_list_wrapper
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
П

Ђtotal

Ѓcount
Є	variables
Ѕ	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
ј

Іtotal

Їcount
Ј
_fn_kwargs
Љ	variables
Њ	keras_api"Ќ
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "mse", "dtype": "float32", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}}
љ

Ћtotal

Ќcount
­
_fn_kwargs
Ў	variables
Џ	keras_api"­
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "mae", "dtype": "float32", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}}
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
:  (2total
:  (2count
0
Ђ0
Ѓ1"
trackable_list_wrapper
.
Є	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
І0
Ї1"
trackable_list_wrapper
.
Љ	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
Ћ0
Ќ1"
trackable_list_wrapper
.
Ў	variables"
_generic_user_object
&:$2 2Adam/dense_27/kernel/m
 : 2Adam/dense_27/bias/m
&:$2@2Adam/dense_28/kernel/m
 :@2Adam/dense_28/bias/m
&:$`2Adam/dense_29/kernel/m
 :2Adam/dense_29/bias/m
3:1	Ќ2"Adam/lstm_18/lstm_cell_18/kernel/m
=:;	KЌ2,Adam/lstm_18/lstm_cell_18/recurrent_kernel/m
-:+Ќ2 Adam/lstm_18/lstm_cell_18/bias/m
/:-	2Adam/gru_9/gru_cell_9/kernel/m
9:7	22(Adam/gru_9/gru_cell_9/recurrent_kernel/m
-:+	2Adam/gru_9/gru_cell_9/bias/m
3:1	KШ2"Adam/lstm_19/lstm_cell_19/kernel/m
=:;	2Ш2,Adam/lstm_19/lstm_cell_19/recurrent_kernel/m
-:+Ш2 Adam/lstm_19/lstm_cell_19/bias/m
&:$2 2Adam/dense_27/kernel/v
 : 2Adam/dense_27/bias/v
&:$2@2Adam/dense_28/kernel/v
 :@2Adam/dense_28/bias/v
&:$`2Adam/dense_29/kernel/v
 :2Adam/dense_29/bias/v
3:1	Ќ2"Adam/lstm_18/lstm_cell_18/kernel/v
=:;	KЌ2,Adam/lstm_18/lstm_cell_18/recurrent_kernel/v
-:+Ќ2 Adam/lstm_18/lstm_cell_18/bias/v
/:-	2Adam/gru_9/gru_cell_9/kernel/v
9:7	22(Adam/gru_9/gru_cell_9/recurrent_kernel/v
-:+	2Adam/gru_9/gru_cell_9/bias/v
3:1	KШ2"Adam/lstm_19/lstm_cell_19/kernel/v
=:;	2Ш2,Adam/lstm_19/lstm_cell_19/recurrent_kernel/v
-:+Ш2 Adam/lstm_19/lstm_cell_19/bias/v
т2п
E__inference_model_9_layer_call_and_return_conditional_losses_44729700
E__inference_model_9_layer_call_and_return_conditional_losses_44729219
E__inference_model_9_layer_call_and_return_conditional_losses_44728522
E__inference_model_9_layer_call_and_return_conditional_losses_44728479Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
я2ь
#__inference__wrapped_model_44725519Ф
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *4Ђ1
/,
input_10џџџџџџџџџџџџџџџџџџ
і2ѓ
*__inference_model_9_layer_call_fn_44728601
*__inference_model_9_layer_call_fn_44729770
*__inference_model_9_layer_call_fn_44728679
*__inference_model_9_layer_call_fn_44729735Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ї2є
E__inference_lstm_18_layer_call_and_return_conditional_losses_44729923
E__inference_lstm_18_layer_call_and_return_conditional_losses_44730076
E__inference_lstm_18_layer_call_and_return_conditional_losses_44730251
E__inference_lstm_18_layer_call_and_return_conditional_losses_44730404е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
*__inference_lstm_18_layer_call_fn_44730087
*__inference_lstm_18_layer_call_fn_44730426
*__inference_lstm_18_layer_call_fn_44730415
*__inference_lstm_18_layer_call_fn_44730098е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ю2Ы
H__inference_dropout_18_layer_call_and_return_conditional_losses_44730438
H__inference_dropout_18_layer_call_and_return_conditional_losses_44730443Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
-__inference_dropout_18_layer_call_fn_44730448
-__inference_dropout_18_layer_call_fn_44730453Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
я2ь
C__inference_gru_9_layer_call_and_return_conditional_losses_44730952
C__inference_gru_9_layer_call_and_return_conditional_losses_44730612
C__inference_gru_9_layer_call_and_return_conditional_losses_44731111
C__inference_gru_9_layer_call_and_return_conditional_losses_44730771е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
(__inference_gru_9_layer_call_fn_44730793
(__inference_gru_9_layer_call_fn_44731122
(__inference_gru_9_layer_call_fn_44730782
(__inference_gru_9_layer_call_fn_44731133е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ї2є
E__inference_lstm_19_layer_call_and_return_conditional_losses_44731439
E__inference_lstm_19_layer_call_and_return_conditional_losses_44731286
E__inference_lstm_19_layer_call_and_return_conditional_losses_44731767
E__inference_lstm_19_layer_call_and_return_conditional_losses_44731614е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
*__inference_lstm_19_layer_call_fn_44731789
*__inference_lstm_19_layer_call_fn_44731778
*__inference_lstm_19_layer_call_fn_44731461
*__inference_lstm_19_layer_call_fn_44731450е
ЬВШ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ю2Ы
H__inference_dropout_19_layer_call_and_return_conditional_losses_44731806
H__inference_dropout_19_layer_call_and_return_conditional_losses_44731801Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
-__inference_dropout_19_layer_call_fn_44731816
-__inference_dropout_19_layer_call_fn_44731811Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
№2э
F__inference_dense_27_layer_call_and_return_conditional_losses_44731827Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
е2в
+__inference_dense_27_layer_call_fn_44731836Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№2э
F__inference_dense_28_layer_call_and_return_conditional_losses_44731847Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
е2в
+__inference_dense_28_layer_call_fn_44731856Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѕ2ђ
K__inference_concatenate_9_layer_call_and_return_conditional_losses_44731863Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
к2з
0__inference_concatenate_9_layer_call_fn_44731869Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№2э
F__inference_dense_29_layer_call_and_return_conditional_losses_44731879Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
е2в
+__inference_dense_29_layer_call_fn_44731888Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЮBЫ
&__inference_signature_wrapper_44728724input_10"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
м2й
J__inference_lstm_cell_18_layer_call_and_return_conditional_losses_44731921
J__inference_lstm_cell_18_layer_call_and_return_conditional_losses_44731954О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
І2Ѓ
/__inference_lstm_cell_18_layer_call_fn_44731971
/__inference_lstm_cell_18_layer_call_fn_44731988О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
и2е
H__inference_gru_cell_9_layer_call_and_return_conditional_losses_44732028
H__inference_gru_cell_9_layer_call_and_return_conditional_losses_44732068О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ђ2
-__inference_gru_cell_9_layer_call_fn_44732082
-__inference_gru_cell_9_layer_call_fn_44732096О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
м2й
J__inference_lstm_cell_19_layer_call_and_return_conditional_losses_44732162
J__inference_lstm_cell_19_layer_call_and_return_conditional_losses_44732129О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
І2Ѓ
/__inference_lstm_cell_19_layer_call_fn_44732196
/__inference_lstm_cell_19_layer_call_fn_44732179О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 Ў
#__inference__wrapped_model_44725519FGHKIJLMN+,12;<>Ђ;
4Ђ1
/,
input_10џџџџџџџџџџџџџџџџџџ
Њ "3Њ0
.
dense_29"
dense_29џџџџџџџџџг
K__inference_concatenate_9_layer_call_and_return_conditional_losses_44731863ZЂW
PЂM
KH
"
inputs/0џџџџџџџџџ 
"
inputs/1џџџџџџџџџ@
Њ "%Ђ"

0џџџџџџџџџ`
 Њ
0__inference_concatenate_9_layer_call_fn_44731869vZЂW
PЂM
KH
"
inputs/0џџџџџџџџџ 
"
inputs/1џџџџџџџџџ@
Њ "џџџџџџџџџ`І
F__inference_dense_27_layer_call_and_return_conditional_losses_44731827\+,/Ђ,
%Ђ"
 
inputsџџџџџџџџџ2
Њ "%Ђ"

0џџџџџџџџџ 
 ~
+__inference_dense_27_layer_call_fn_44731836O+,/Ђ,
%Ђ"
 
inputsџџџџџџџџџ2
Њ "џџџџџџџџџ І
F__inference_dense_28_layer_call_and_return_conditional_losses_44731847\12/Ђ,
%Ђ"
 
inputsџџџџџџџџџ2
Њ "%Ђ"

0џџџџџџџџџ@
 ~
+__inference_dense_28_layer_call_fn_44731856O12/Ђ,
%Ђ"
 
inputsџџџџџџџџџ2
Њ "џџџџџџџџџ@І
F__inference_dense_29_layer_call_and_return_conditional_losses_44731879\;</Ђ,
%Ђ"
 
inputsџџџџџџџџџ`
Њ "%Ђ"

0џџџџџџџџџ
 ~
+__inference_dense_29_layer_call_fn_44731888O;</Ђ,
%Ђ"
 
inputsџџџџџџџџџ`
Њ "џџџџџџџџџТ
H__inference_dropout_18_layer_call_and_return_conditional_losses_44730438v@Ђ=
6Ђ3
-*
inputsџџџџџџџџџџџџџџџџџџK
p
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџK
 Т
H__inference_dropout_18_layer_call_and_return_conditional_losses_44730443v@Ђ=
6Ђ3
-*
inputsџџџџџџџџџџџџџџџџџџK
p 
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџK
 
-__inference_dropout_18_layer_call_fn_44730448i@Ђ=
6Ђ3
-*
inputsџџџџџџџџџџџџџџџџџџK
p
Њ "%"џџџџџџџџџџџџџџџџџџK
-__inference_dropout_18_layer_call_fn_44730453i@Ђ=
6Ђ3
-*
inputsџџџџџџџџџџџџџџџџџџK
p 
Њ "%"џџџџџџџџџџџџџџџџџџKЈ
H__inference_dropout_19_layer_call_and_return_conditional_losses_44731801\3Ђ0
)Ђ&
 
inputsџџџџџџџџџ2
p
Њ "%Ђ"

0џџџџџџџџџ2
 Ј
H__inference_dropout_19_layer_call_and_return_conditional_losses_44731806\3Ђ0
)Ђ&
 
inputsџџџџџџџџџ2
p 
Њ "%Ђ"

0џџџџџџџџџ2
 
-__inference_dropout_19_layer_call_fn_44731811O3Ђ0
)Ђ&
 
inputsџџџџџџџџџ2
p
Њ "џџџџџџџџџ2
-__inference_dropout_19_layer_call_fn_44731816O3Ђ0
)Ђ&
 
inputsџџџџџџџџџ2
p 
Њ "џџџџџџџџџ2Н
C__inference_gru_9_layer_call_and_return_conditional_losses_44730612vKIJHЂE
>Ђ;
-*
inputsџџџџџџџџџџџџџџџџџџ

 
p

 
Њ "%Ђ"

0џџџџџџџџџ2
 Н
C__inference_gru_9_layer_call_and_return_conditional_losses_44730771vKIJHЂE
>Ђ;
-*
inputsџџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "%Ђ"

0џџџџџџџџџ2
 Ф
C__inference_gru_9_layer_call_and_return_conditional_losses_44730952}KIJOЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ "%Ђ"

0џџџџџџџџџ2
 Ф
C__inference_gru_9_layer_call_and_return_conditional_losses_44731111}KIJOЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "%Ђ"

0џџџџџџџџџ2
 
(__inference_gru_9_layer_call_fn_44730782iKIJHЂE
>Ђ;
-*
inputsџџџџџџџџџџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџ2
(__inference_gru_9_layer_call_fn_44730793iKIJHЂE
>Ђ;
-*
inputsџџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџ2
(__inference_gru_9_layer_call_fn_44731122pKIJOЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџ2
(__inference_gru_9_layer_call_fn_44731133pKIJOЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџ2
H__inference_gru_cell_9_layer_call_and_return_conditional_losses_44732028ЗKIJ\ЂY
RЂO
 
inputsџџџџџџџџџ
'Ђ$
"
states/0џџџџџџџџџ2
p
Њ "RЂO
HЂE

0/0џџџџџџџџџ2
$!

0/1/0џџџџџџџџџ2
 
H__inference_gru_cell_9_layer_call_and_return_conditional_losses_44732068ЗKIJ\ЂY
RЂO
 
inputsџџџџџџџџџ
'Ђ$
"
states/0џџџџџџџџџ2
p 
Њ "RЂO
HЂE

0/0џџџџџџџџџ2
$!

0/1/0џџџџџџџџџ2
 л
-__inference_gru_cell_9_layer_call_fn_44732082ЉKIJ\ЂY
RЂO
 
inputsџџџџџџџџџ
'Ђ$
"
states/0џџџџџџџџџ2
p
Њ "DЂA

0џџџџџџџџџ2
"

1/0џџџџџџџџџ2л
-__inference_gru_cell_9_layer_call_fn_44732096ЉKIJ\ЂY
RЂO
 
inputsџџџџџџџџџ
'Ђ$
"
states/0џџџџџџџџџ2
p 
Њ "DЂA

0џџџџџџџџџ2
"

1/0џџџџџџџџџ2Э
E__inference_lstm_18_layer_call_and_return_conditional_losses_44729923FGHHЂE
>Ђ;
-*
inputsџџџџџџџџџџџџџџџџџџ

 
p

 
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџK
 Э
E__inference_lstm_18_layer_call_and_return_conditional_losses_44730076FGHHЂE
>Ђ;
-*
inputsџџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџK
 д
E__inference_lstm_18_layer_call_and_return_conditional_losses_44730251FGHOЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџK
 д
E__inference_lstm_18_layer_call_and_return_conditional_losses_44730404FGHOЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџK
 Є
*__inference_lstm_18_layer_call_fn_44730087vFGHHЂE
>Ђ;
-*
inputsџџџџџџџџџџџџџџџџџџ

 
p

 
Њ "%"џџџџџџџџџџџџџџџџџџKЄ
*__inference_lstm_18_layer_call_fn_44730098vFGHHЂE
>Ђ;
-*
inputsџџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "%"џџџџџџџџџџџџџџџџџџKЋ
*__inference_lstm_18_layer_call_fn_44730415}FGHOЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ "%"џџџџџџџџџџџџџџџџџџKЋ
*__inference_lstm_18_layer_call_fn_44730426}FGHOЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "%"џџџџџџџџџџџџџџџџџџKЦ
E__inference_lstm_19_layer_call_and_return_conditional_losses_44731286}LMNOЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџK

 
p

 
Њ "%Ђ"

0џџџџџџџџџ2
 Ц
E__inference_lstm_19_layer_call_and_return_conditional_losses_44731439}LMNOЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџK

 
p 

 
Њ "%Ђ"

0џџџџџџџџџ2
 П
E__inference_lstm_19_layer_call_and_return_conditional_losses_44731614vLMNHЂE
>Ђ;
-*
inputsџџџџџџџџџџџџџџџџџџK

 
p

 
Њ "%Ђ"

0џџџџџџџџџ2
 П
E__inference_lstm_19_layer_call_and_return_conditional_losses_44731767vLMNHЂE
>Ђ;
-*
inputsџџџџџџџџџџџџџџџџџџK

 
p 

 
Њ "%Ђ"

0џџџџџџџџџ2
 
*__inference_lstm_19_layer_call_fn_44731450pLMNOЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџK

 
p

 
Њ "џџџџџџџџџ2
*__inference_lstm_19_layer_call_fn_44731461pLMNOЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџK

 
p 

 
Њ "џџџџџџџџџ2
*__inference_lstm_19_layer_call_fn_44731778iLMNHЂE
>Ђ;
-*
inputsџџџџџџџџџџџџџџџџџџK

 
p

 
Њ "џџџџџџџџџ2
*__inference_lstm_19_layer_call_fn_44731789iLMNHЂE
>Ђ;
-*
inputsџџџџџџџџџџџџџџџџџџK

 
p 

 
Њ "џџџџџџџџџ2Ь
J__inference_lstm_cell_18_layer_call_and_return_conditional_losses_44731921§FGHЂ}
vЂs
 
inputsџџџџџџџџџ
KЂH
"
states/0џџџџџџџџџK
"
states/1џџџџџџџџџK
p
Њ "sЂp
iЂf

0/0џџџџџџџџџK
EB

0/1/0џџџџџџџџџK

0/1/1џџџџџџџџџK
 Ь
J__inference_lstm_cell_18_layer_call_and_return_conditional_losses_44731954§FGHЂ}
vЂs
 
inputsџџџџџџџџџ
KЂH
"
states/0џџџџџџџџџK
"
states/1џџџџџџџџџK
p 
Њ "sЂp
iЂf

0/0џџџџџџџџџK
EB

0/1/0џџџџџџџџџK

0/1/1џџџџџџџџџK
 Ё
/__inference_lstm_cell_18_layer_call_fn_44731971эFGHЂ}
vЂs
 
inputsџџџџџџџџџ
KЂH
"
states/0џџџџџџџџџK
"
states/1џџџџџџџџџK
p
Њ "cЂ`

0џџџџџџџџџK
A>

1/0џџџџџџџџџK

1/1џџџџџџџџџKЁ
/__inference_lstm_cell_18_layer_call_fn_44731988эFGHЂ}
vЂs
 
inputsџџџџџџџџџ
KЂH
"
states/0џџџџџџџџџK
"
states/1џџџџџџџџџK
p 
Њ "cЂ`

0џџџџџџџџџK
A>

1/0џџџџџџџџџK

1/1џџџџџџџџџKЬ
J__inference_lstm_cell_19_layer_call_and_return_conditional_losses_44732129§LMNЂ}
vЂs
 
inputsџџџџџџџџџK
KЂH
"
states/0џџџџџџџџџ2
"
states/1џџџџџџџџџ2
p
Њ "sЂp
iЂf

0/0џџџџџџџџџ2
EB

0/1/0џџџџџџџџџ2

0/1/1џџџџџџџџџ2
 Ь
J__inference_lstm_cell_19_layer_call_and_return_conditional_losses_44732162§LMNЂ}
vЂs
 
inputsџџџџџџџџџK
KЂH
"
states/0џџџџџџџџџ2
"
states/1џџџџџџџџџ2
p 
Њ "sЂp
iЂf

0/0џџџџџџџџџ2
EB

0/1/0џџџџџџџџџ2

0/1/1џџџџџџџџџ2
 Ё
/__inference_lstm_cell_19_layer_call_fn_44732179эLMNЂ}
vЂs
 
inputsџџџџџџџџџK
KЂH
"
states/0џџџџџџџџџ2
"
states/1џџџџџџџџџ2
p
Њ "cЂ`

0џџџџџџџџџ2
A>

1/0џџџџџџџџџ2

1/1џџџџџџџџџ2Ё
/__inference_lstm_cell_19_layer_call_fn_44732196эLMNЂ}
vЂs
 
inputsџџџџџџџџџK
KЂH
"
states/0џџџџџџџџџ2
"
states/1џџџџџџџџџ2
p 
Њ "cЂ`

0џџџџџџџџџ2
A>

1/0џџџџџџџџџ2

1/1џџџџџџџџџ2Ъ
E__inference_model_9_layer_call_and_return_conditional_losses_44728479FGHKIJLMN+,12;<FЂC
<Ђ9
/,
input_10џџџџџџџџџџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 Ъ
E__inference_model_9_layer_call_and_return_conditional_losses_44728522FGHKIJLMN+,12;<FЂC
<Ђ9
/,
input_10џџџџџџџџџџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 Ч
E__inference_model_9_layer_call_and_return_conditional_losses_44729219~FGHKIJLMN+,12;<DЂA
:Ђ7
-*
inputsџџџџџџџџџџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 Ч
E__inference_model_9_layer_call_and_return_conditional_losses_44729700~FGHKIJLMN+,12;<DЂA
:Ђ7
-*
inputsџџџџџџџџџџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 Ё
*__inference_model_9_layer_call_fn_44728601sFGHKIJLMN+,12;<FЂC
<Ђ9
/,
input_10џџџџџџџџџџџџџџџџџџ
p

 
Њ "џџџџџџџџџЁ
*__inference_model_9_layer_call_fn_44728679sFGHKIJLMN+,12;<FЂC
<Ђ9
/,
input_10џџџџџџџџџџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
*__inference_model_9_layer_call_fn_44729735qFGHKIJLMN+,12;<DЂA
:Ђ7
-*
inputsџџџџџџџџџџџџџџџџџџ
p

 
Њ "џџџџџџџџџ
*__inference_model_9_layer_call_fn_44729770qFGHKIJLMN+,12;<DЂA
:Ђ7
-*
inputsџџџџџџџџџџџџџџџџџџ
p 

 
Њ "џџџџџџџџџН
&__inference_signature_wrapper_44728724FGHKIJLMN+,12;<JЂG
Ђ 
@Њ=
;
input_10/,
input_10џџџџџџџџџџџџџџџџџџ"3Њ0
.
dense_29"
dense_29џџџџџџџџџ