��
��
B
AddV2
x"T
y"T
z"T"
Ttype:
2	��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
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
delete_old_dirsbool(�
=
Mul
x"T
y"T
z"T"
Ttype:
2	�
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
dtypetype�
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
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
�
SplitV

value"T
size_splits"Tlen
	split_dim
output"T*	num_split"
	num_splitint(0"	
Ttype"
Tlentype0	:
2	
�
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
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
�
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
�
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type
output_handle"
element_dtypetype"

shape_typetype:
2	
�
TensorListReserve
element_shape"
shape_type
num_elements

handle"
element_dtypetype"

shape_typetype:
2	
�
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint���������
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
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
�
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
�"serve*2.4.12unknown8��
z
dense_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Kd* 
shared_namedense_18/kernel
s
#dense_18/kernel/Read/ReadVariableOpReadVariableOpdense_18/kernel*
_output_shapes

:Kd*
dtype0
r
dense_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_18/bias
k
!dense_18/bias/Read/ReadVariableOpReadVariableOpdense_18/bias*
_output_shapes
:d*
dtype0
z
dense_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d* 
shared_namedense_19/kernel
s
#dense_19/kernel/Read/ReadVariableOpReadVariableOpdense_19/kernel*
_output_shapes

:d*
dtype0
r
dense_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_19/bias
k
!dense_19/bias/Read/ReadVariableOpReadVariableOpdense_19/bias*
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
�
gru_9/gru_cell_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*(
shared_namegru_9/gru_cell_9/kernel
�
+gru_9/gru_cell_9/kernel/Read/ReadVariableOpReadVariableOpgru_9/gru_cell_9/kernel*
_output_shapes
:	�*
dtype0
�
!gru_9/gru_cell_9/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	K�*2
shared_name#!gru_9/gru_cell_9/recurrent_kernel
�
5gru_9/gru_cell_9/recurrent_kernel/Read/ReadVariableOpReadVariableOp!gru_9/gru_cell_9/recurrent_kernel*
_output_shapes
:	K�*
dtype0
�
gru_9/gru_cell_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*&
shared_namegru_9/gru_cell_9/bias
�
)gru_9/gru_cell_9/bias/Read/ReadVariableOpReadVariableOpgru_9/gru_cell_9/bias*
_output_shapes
:	�*
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
�
Adam/dense_18/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Kd*'
shared_nameAdam/dense_18/kernel/m
�
*Adam/dense_18/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_18/kernel/m*
_output_shapes

:Kd*
dtype0
�
Adam/dense_18/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameAdam/dense_18/bias/m
y
(Adam/dense_18/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_18/bias/m*
_output_shapes
:d*
dtype0
�
Adam/dense_19/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*'
shared_nameAdam/dense_19/kernel/m
�
*Adam/dense_19/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_19/kernel/m*
_output_shapes

:d*
dtype0
�
Adam/dense_19/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_19/bias/m
y
(Adam/dense_19/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_19/bias/m*
_output_shapes
:*
dtype0
�
Adam/gru_9/gru_cell_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*/
shared_name Adam/gru_9/gru_cell_9/kernel/m
�
2Adam/gru_9/gru_cell_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/gru_9/gru_cell_9/kernel/m*
_output_shapes
:	�*
dtype0
�
(Adam/gru_9/gru_cell_9/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	K�*9
shared_name*(Adam/gru_9/gru_cell_9/recurrent_kernel/m
�
<Adam/gru_9/gru_cell_9/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp(Adam/gru_9/gru_cell_9/recurrent_kernel/m*
_output_shapes
:	K�*
dtype0
�
Adam/gru_9/gru_cell_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*-
shared_nameAdam/gru_9/gru_cell_9/bias/m
�
0Adam/gru_9/gru_cell_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/gru_9/gru_cell_9/bias/m*
_output_shapes
:	�*
dtype0
�
Adam/dense_18/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Kd*'
shared_nameAdam/dense_18/kernel/v
�
*Adam/dense_18/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_18/kernel/v*
_output_shapes

:Kd*
dtype0
�
Adam/dense_18/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameAdam/dense_18/bias/v
y
(Adam/dense_18/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_18/bias/v*
_output_shapes
:d*
dtype0
�
Adam/dense_19/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*'
shared_nameAdam/dense_19/kernel/v
�
*Adam/dense_19/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_19/kernel/v*
_output_shapes

:d*
dtype0
�
Adam/dense_19/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_19/bias/v
y
(Adam/dense_19/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_19/bias/v*
_output_shapes
:*
dtype0
�
Adam/gru_9/gru_cell_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*/
shared_name Adam/gru_9/gru_cell_9/kernel/v
�
2Adam/gru_9/gru_cell_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/gru_9/gru_cell_9/kernel/v*
_output_shapes
:	�*
dtype0
�
(Adam/gru_9/gru_cell_9/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	K�*9
shared_name*(Adam/gru_9/gru_cell_9/recurrent_kernel/v
�
<Adam/gru_9/gru_cell_9/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp(Adam/gru_9/gru_cell_9/recurrent_kernel/v*
_output_shapes
:	K�*
dtype0
�
Adam/gru_9/gru_cell_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*-
shared_nameAdam/gru_9/gru_cell_9/bias/v
�
0Adam/gru_9/gru_cell_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/gru_9/gru_cell_9/bias/v*
_output_shapes
:	�*
dtype0

NoOpNoOp
�-
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�-
value�-B�- B�-
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api
	
signatures
l

cell

state_spec
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
�
iter

beta_1

beta_2
	decay
 learning_ratemSmTmUmV!mW"mX#mYvZv[v\v]!v^"v_#v`
1
!0
"1
#2
3
4
5
6
 
1
!0
"1
#2
3
4
5
6
�
$layer_metrics

%layers
&layer_regularization_losses
trainable_variables
regularization_losses
'non_trainable_variables
(metrics
	variables
 
~

!kernel
"recurrent_kernel
#bias
)trainable_variables
*regularization_losses
+	variables
,	keras_api
 

!0
"1
#2
 

!0
"1
#2
�
-layer_metrics

.layers
/layer_regularization_losses
trainable_variables
regularization_losses
0non_trainable_variables
1metrics

2states
	variables
[Y
VARIABLE_VALUEdense_18/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_18/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
3layer_metrics

4layers
5layer_regularization_losses
trainable_variables
regularization_losses
6non_trainable_variables
7metrics
	variables
[Y
VARIABLE_VALUEdense_19/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_19/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
8layer_metrics

9layers
:layer_regularization_losses
trainable_variables
regularization_losses
;non_trainable_variables
<metrics
	variables
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
][
VARIABLE_VALUEgru_9/gru_cell_9/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE!gru_9/gru_cell_9/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEgru_9/gru_cell_9/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2
 
 

=0
>1
?2

!0
"1
#2
 

!0
"1
#2
�
@layer_metrics

Alayers
Blayer_regularization_losses
)trainable_variables
*regularization_losses
Cnon_trainable_variables
Dmetrics
+	variables
 


0
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
4
	Etotal
	Fcount
G	variables
H	keras_api
D
	Itotal
	Jcount
K
_fn_kwargs
L	variables
M	keras_api
D
	Ntotal
	Ocount
P
_fn_kwargs
Q	variables
R	keras_api
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

E0
F1

G	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

I0
J1

L	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

N0
O1

Q	variables
~|
VARIABLE_VALUEAdam/dense_18/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_18/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_19/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_19/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/gru_9/gru_cell_9/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE(Adam/gru_9/gru_cell_9/recurrent_kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/gru_9/gru_cell_9/bias/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_18/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_18/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_19/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_19/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/gru_9/gru_cell_9/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE(Adam/gru_9/gru_cell_9/recurrent_kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/gru_9/gru_cell_9/bias/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_gru_9_inputPlaceholder*+
_output_shapes
:���������*
dtype0* 
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_gru_9_inputgru_9/gru_cell_9/biasgru_9/gru_cell_9/kernel!gru_9/gru_cell_9/recurrent_kerneldense_18/kerneldense_18/biasdense_19/kerneldense_19/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8� */
f*R(
&__inference_signature_wrapper_27549570
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_18/kernel/Read/ReadVariableOp!dense_18/bias/Read/ReadVariableOp#dense_19/kernel/Read/ReadVariableOp!dense_19/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp+gru_9/gru_cell_9/kernel/Read/ReadVariableOp5gru_9/gru_cell_9/recurrent_kernel/Read/ReadVariableOp)gru_9/gru_cell_9/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp*Adam/dense_18/kernel/m/Read/ReadVariableOp(Adam/dense_18/bias/m/Read/ReadVariableOp*Adam/dense_19/kernel/m/Read/ReadVariableOp(Adam/dense_19/bias/m/Read/ReadVariableOp2Adam/gru_9/gru_cell_9/kernel/m/Read/ReadVariableOp<Adam/gru_9/gru_cell_9/recurrent_kernel/m/Read/ReadVariableOp0Adam/gru_9/gru_cell_9/bias/m/Read/ReadVariableOp*Adam/dense_18/kernel/v/Read/ReadVariableOp(Adam/dense_18/bias/v/Read/ReadVariableOp*Adam/dense_19/kernel/v/Read/ReadVariableOp(Adam/dense_19/bias/v/Read/ReadVariableOp2Adam/gru_9/gru_cell_9/kernel/v/Read/ReadVariableOp<Adam/gru_9/gru_cell_9/recurrent_kernel/v/Read/ReadVariableOp0Adam/gru_9/gru_cell_9/bias/v/Read/ReadVariableOpConst*-
Tin&
$2"	*
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
GPU2*0J 8� **
f%R#
!__inference__traced_save_27550898
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_18/kerneldense_18/biasdense_19/kerneldense_19/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rategru_9/gru_cell_9/kernel!gru_9/gru_cell_9/recurrent_kernelgru_9/gru_cell_9/biastotalcounttotal_1count_1total_2count_2Adam/dense_18/kernel/mAdam/dense_18/bias/mAdam/dense_19/kernel/mAdam/dense_19/bias/mAdam/gru_9/gru_cell_9/kernel/m(Adam/gru_9/gru_cell_9/recurrent_kernel/mAdam/gru_9/gru_cell_9/bias/mAdam/dense_18/kernel/vAdam/dense_18/bias/vAdam/dense_19/kernel/vAdam/dense_19/bias/vAdam/gru_9/gru_cell_9/kernel/v(Adam/gru_9/gru_cell_9/recurrent_kernel/vAdam/gru_9/gru_cell_9/bias/v*,
Tin%
#2!*
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
GPU2*0J 8� *-
f(R&
$__inference__traced_restore_27551004Đ
�
�
while_cond_27550360
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_27550360___redundant_placeholder06
2while_while_cond_27550360___redundant_placeholder16
2while_while_cond_27550360___redundant_placeholder26
2while_while_cond_27550360___redundant_placeholder3
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
-: : : : :���������K: ::::: 
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
:���������K:

_output_shapes
: :

_output_shapes
:
�Z
�
C__inference_gru_9_layer_call_and_return_conditional_losses_27550451

inputs&
"gru_cell_9_readvariableop_resource-
)gru_cell_9_matmul_readvariableop_resource/
+gru_cell_9_matmul_1_readvariableop_resource
identity�� gru_cell_9/MatMul/ReadVariableOp�"gru_cell_9/MatMul_1/ReadVariableOp�gru_cell_9/ReadVariableOp�whileD
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
strided_slice/stack_2�
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
B :�2
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
zeros/packed/1�
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
:���������K2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������2
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
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
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
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
gru_cell_9/ReadVariableOpReadVariableOp"gru_cell_9_readvariableop_resource*
_output_shapes
:	�*
dtype02
gru_cell_9/ReadVariableOp�
gru_cell_9/unstackUnpack!gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2
gru_cell_9/unstack�
 gru_cell_9/MatMul/ReadVariableOpReadVariableOp)gru_cell_9_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02"
 gru_cell_9/MatMul/ReadVariableOp�
gru_cell_9/MatMulMatMulstrided_slice_2:output:0(gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
gru_cell_9/MatMul�
gru_cell_9/BiasAddBiasAddgru_cell_9/MatMul:product:0gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:����������2
gru_cell_9/BiasAddf
gru_cell_9/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_9/Const�
gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
gru_cell_9/split/split_dim�
gru_cell_9/splitSplit#gru_cell_9/split/split_dim:output:0gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
gru_cell_9/split�
"gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_9_matmul_1_readvariableop_resource*
_output_shapes
:	K�*
dtype02$
"gru_cell_9/MatMul_1/ReadVariableOp�
gru_cell_9/MatMul_1MatMulzeros:output:0*gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
gru_cell_9/MatMul_1�
gru_cell_9/BiasAdd_1BiasAddgru_cell_9/MatMul_1:product:0gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:����������2
gru_cell_9/BiasAdd_1}
gru_cell_9/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"K   K   ����2
gru_cell_9/Const_1�
gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
gru_cell_9/split_1/split_dim�
gru_cell_9/split_1SplitVgru_cell_9/BiasAdd_1:output:0gru_cell_9/Const_1:output:0%gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
gru_cell_9/split_1�
gru_cell_9/addAddV2gru_cell_9/split:output:0gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/addy
gru_cell_9/SigmoidSigmoidgru_cell_9/add:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/Sigmoid�
gru_cell_9/add_1AddV2gru_cell_9/split:output:1gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:���������K2
gru_cell_9/add_1
gru_cell_9/Sigmoid_1Sigmoidgru_cell_9/add_1:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/Sigmoid_1�
gru_cell_9/mulMulgru_cell_9/Sigmoid_1:y:0gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:���������K2
gru_cell_9/mul�
gru_cell_9/add_2AddV2gru_cell_9/split:output:2gru_cell_9/mul:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/add_2r
gru_cell_9/ReluRelugru_cell_9/add_2:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/Relu�
gru_cell_9/mul_1Mulgru_cell_9/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/mul_1i
gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
gru_cell_9/sub/x�
gru_cell_9/subSubgru_cell_9/sub/x:output:0gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/sub�
gru_cell_9/mul_2Mulgru_cell_9/sub:z:0gru_cell_9/Relu:activations:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/mul_2�
gru_cell_9/add_3AddV2gru_cell_9/mul_1:z:0gru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/add_3�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����K   2
TensorArrayV2_1/element_shape�
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
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_9_readvariableop_resource)gru_cell_9_matmul_readvariableop_resource+gru_cell_9_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������K: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_27550361*
condR
while_cond_27550360*8
output_shapes'
%: : : : :���������K: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����K   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������K*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
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
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������K*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������K2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime�
IdentityIdentitystrided_slice_3:output:0!^gru_cell_9/MatMul/ReadVariableOp#^gru_cell_9/MatMul_1/ReadVariableOp^gru_cell_9/ReadVariableOp^while*
T0*'
_output_shapes
:���������K2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������:::2D
 gru_cell_9/MatMul/ReadVariableOp gru_cell_9/MatMul/ReadVariableOp2H
"gru_cell_9/MatMul_1/ReadVariableOp"gru_cell_9/MatMul_1/ReadVariableOp26
gru_cell_9/ReadVariableOpgru_cell_9/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
J__inference_sequential_9_layer_call_and_return_conditional_losses_27549484

inputs
gru_9_27549466
gru_9_27549468
gru_9_27549470
dense_18_27549473
dense_18_27549475
dense_19_27549478
dense_19_27549480
identity�� dense_18/StatefulPartitionedCall� dense_19/StatefulPartitionedCall�gru_9/StatefulPartitionedCall�
gru_9/StatefulPartitionedCallStatefulPartitionedCallinputsgru_9_27549466gru_9_27549468gru_9_27549470*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������K*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_gru_9_layer_call_and_return_conditional_losses_275491962
gru_9/StatefulPartitionedCall�
 dense_18/StatefulPartitionedCallStatefulPartitionedCall&gru_9/StatefulPartitionedCall:output:0dense_18_27549473dense_18_27549475*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_18_layer_call_and_return_conditional_losses_275493962"
 dense_18/StatefulPartitionedCall�
 dense_19/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0dense_19_27549478dense_19_27549480*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_19_layer_call_and_return_conditional_losses_275494222"
 dense_19/StatefulPartitionedCall�
IdentityIdentity)dense_19/StatefulPartitionedCall:output:0!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall^gru_9/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������:::::::2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2>
gru_9/StatefulPartitionedCallgru_9/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
(__inference_gru_9_layer_call_fn_27550621

inputs
unknown
	unknown_0
	unknown_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������K*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_gru_9_layer_call_and_return_conditional_losses_275491962
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������K2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
/__inference_sequential_9_layer_call_fn_27549952

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_sequential_9_layer_call_and_return_conditional_losses_275495242
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������:::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
while_cond_27549264
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_27549264___redundant_placeholder06
2while_while_cond_27549264___redundant_placeholder16
2while_while_cond_27549264___redundant_placeholder26
2while_while_cond_27549264___redundant_placeholder3
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
-: : : : :���������K: ::::: 
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
:���������K:

_output_shapes
: :

_output_shapes
:
�
�
&__inference_signature_wrapper_27549570
gru_9_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallgru_9_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8� *,
f'R%
#__inference__wrapped_model_275484722
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������:::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:���������
%
_user_specified_namegru_9_input
�!
�
while_body_27548843
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_gru_cell_9_27548865_0
while_gru_cell_9_27548867_0
while_gru_cell_9_27548869_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_gru_cell_9_27548865
while_gru_cell_9_27548867
while_gru_cell_9_27548869��(while/gru_cell_9/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
(while/gru_cell_9/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_9_27548865_0while_gru_cell_9_27548867_0while_gru_cell_9_27548869_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������K:���������K*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_gru_cell_9_layer_call_and_return_conditional_losses_275485442*
(while/gru_cell_9/StatefulPartitionedCall�
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
while/add_1�
while/IdentityIdentitywhile/add_1:z:0)^while/gru_cell_9/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations)^while/gru_cell_9/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0)^while/gru_cell_9/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/gru_cell_9/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity1while/gru_cell_9/StatefulPartitionedCall:output:1)^while/gru_cell_9/StatefulPartitionedCall*
T0*'
_output_shapes
:���������K2
while/Identity_4"8
while_gru_cell_9_27548865while_gru_cell_9_27548865_0"8
while_gru_cell_9_27548867while_gru_cell_9_27548867_0"8
while_gru_cell_9_27548869while_gru_cell_9_27548869_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :���������K: : :::2T
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
:���������K:

_output_shapes
: :

_output_shapes
: 
�Z
�
C__inference_gru_9_layer_call_and_return_conditional_losses_27549355

inputs&
"gru_cell_9_readvariableop_resource-
)gru_cell_9_matmul_readvariableop_resource/
+gru_cell_9_matmul_1_readvariableop_resource
identity�� gru_cell_9/MatMul/ReadVariableOp�"gru_cell_9/MatMul_1/ReadVariableOp�gru_cell_9/ReadVariableOp�whileD
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
strided_slice/stack_2�
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
B :�2
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
zeros/packed/1�
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
:���������K2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������2
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
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
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
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
gru_cell_9/ReadVariableOpReadVariableOp"gru_cell_9_readvariableop_resource*
_output_shapes
:	�*
dtype02
gru_cell_9/ReadVariableOp�
gru_cell_9/unstackUnpack!gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2
gru_cell_9/unstack�
 gru_cell_9/MatMul/ReadVariableOpReadVariableOp)gru_cell_9_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02"
 gru_cell_9/MatMul/ReadVariableOp�
gru_cell_9/MatMulMatMulstrided_slice_2:output:0(gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
gru_cell_9/MatMul�
gru_cell_9/BiasAddBiasAddgru_cell_9/MatMul:product:0gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:����������2
gru_cell_9/BiasAddf
gru_cell_9/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_9/Const�
gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
gru_cell_9/split/split_dim�
gru_cell_9/splitSplit#gru_cell_9/split/split_dim:output:0gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
gru_cell_9/split�
"gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_9_matmul_1_readvariableop_resource*
_output_shapes
:	K�*
dtype02$
"gru_cell_9/MatMul_1/ReadVariableOp�
gru_cell_9/MatMul_1MatMulzeros:output:0*gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
gru_cell_9/MatMul_1�
gru_cell_9/BiasAdd_1BiasAddgru_cell_9/MatMul_1:product:0gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:����������2
gru_cell_9/BiasAdd_1}
gru_cell_9/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"K   K   ����2
gru_cell_9/Const_1�
gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
gru_cell_9/split_1/split_dim�
gru_cell_9/split_1SplitVgru_cell_9/BiasAdd_1:output:0gru_cell_9/Const_1:output:0%gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
gru_cell_9/split_1�
gru_cell_9/addAddV2gru_cell_9/split:output:0gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/addy
gru_cell_9/SigmoidSigmoidgru_cell_9/add:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/Sigmoid�
gru_cell_9/add_1AddV2gru_cell_9/split:output:1gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:���������K2
gru_cell_9/add_1
gru_cell_9/Sigmoid_1Sigmoidgru_cell_9/add_1:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/Sigmoid_1�
gru_cell_9/mulMulgru_cell_9/Sigmoid_1:y:0gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:���������K2
gru_cell_9/mul�
gru_cell_9/add_2AddV2gru_cell_9/split:output:2gru_cell_9/mul:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/add_2r
gru_cell_9/ReluRelugru_cell_9/add_2:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/Relu�
gru_cell_9/mul_1Mulgru_cell_9/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/mul_1i
gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
gru_cell_9/sub/x�
gru_cell_9/subSubgru_cell_9/sub/x:output:0gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/sub�
gru_cell_9/mul_2Mulgru_cell_9/sub:z:0gru_cell_9/Relu:activations:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/mul_2�
gru_cell_9/add_3AddV2gru_cell_9/mul_1:z:0gru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/add_3�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����K   2
TensorArrayV2_1/element_shape�
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
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_9_readvariableop_resource)gru_cell_9_matmul_readvariableop_resource+gru_cell_9_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������K: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_27549265*
condR
while_cond_27549264*8
output_shapes'
%: : : : :���������K: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����K   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������K*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
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
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������K*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������K2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime�
IdentityIdentitystrided_slice_3:output:0!^gru_cell_9/MatMul/ReadVariableOp#^gru_cell_9/MatMul_1/ReadVariableOp^gru_cell_9/ReadVariableOp^while*
T0*'
_output_shapes
:���������K2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������:::2D
 gru_cell_9/MatMul/ReadVariableOp gru_cell_9/MatMul/ReadVariableOp2H
"gru_cell_9/MatMul_1/ReadVariableOp"gru_cell_9/MatMul_1/ReadVariableOp26
gru_cell_9/ReadVariableOpgru_cell_9/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�Z
�
C__inference_gru_9_layer_call_and_return_conditional_losses_27549196

inputs&
"gru_cell_9_readvariableop_resource-
)gru_cell_9_matmul_readvariableop_resource/
+gru_cell_9_matmul_1_readvariableop_resource
identity�� gru_cell_9/MatMul/ReadVariableOp�"gru_cell_9/MatMul_1/ReadVariableOp�gru_cell_9/ReadVariableOp�whileD
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
strided_slice/stack_2�
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
B :�2
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
zeros/packed/1�
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
:���������K2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������2
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
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
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
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
gru_cell_9/ReadVariableOpReadVariableOp"gru_cell_9_readvariableop_resource*
_output_shapes
:	�*
dtype02
gru_cell_9/ReadVariableOp�
gru_cell_9/unstackUnpack!gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2
gru_cell_9/unstack�
 gru_cell_9/MatMul/ReadVariableOpReadVariableOp)gru_cell_9_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02"
 gru_cell_9/MatMul/ReadVariableOp�
gru_cell_9/MatMulMatMulstrided_slice_2:output:0(gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
gru_cell_9/MatMul�
gru_cell_9/BiasAddBiasAddgru_cell_9/MatMul:product:0gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:����������2
gru_cell_9/BiasAddf
gru_cell_9/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_9/Const�
gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
gru_cell_9/split/split_dim�
gru_cell_9/splitSplit#gru_cell_9/split/split_dim:output:0gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
gru_cell_9/split�
"gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_9_matmul_1_readvariableop_resource*
_output_shapes
:	K�*
dtype02$
"gru_cell_9/MatMul_1/ReadVariableOp�
gru_cell_9/MatMul_1MatMulzeros:output:0*gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
gru_cell_9/MatMul_1�
gru_cell_9/BiasAdd_1BiasAddgru_cell_9/MatMul_1:product:0gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:����������2
gru_cell_9/BiasAdd_1}
gru_cell_9/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"K   K   ����2
gru_cell_9/Const_1�
gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
gru_cell_9/split_1/split_dim�
gru_cell_9/split_1SplitVgru_cell_9/BiasAdd_1:output:0gru_cell_9/Const_1:output:0%gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
gru_cell_9/split_1�
gru_cell_9/addAddV2gru_cell_9/split:output:0gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/addy
gru_cell_9/SigmoidSigmoidgru_cell_9/add:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/Sigmoid�
gru_cell_9/add_1AddV2gru_cell_9/split:output:1gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:���������K2
gru_cell_9/add_1
gru_cell_9/Sigmoid_1Sigmoidgru_cell_9/add_1:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/Sigmoid_1�
gru_cell_9/mulMulgru_cell_9/Sigmoid_1:y:0gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:���������K2
gru_cell_9/mul�
gru_cell_9/add_2AddV2gru_cell_9/split:output:2gru_cell_9/mul:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/add_2r
gru_cell_9/ReluRelugru_cell_9/add_2:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/Relu�
gru_cell_9/mul_1Mulgru_cell_9/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/mul_1i
gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
gru_cell_9/sub/x�
gru_cell_9/subSubgru_cell_9/sub/x:output:0gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/sub�
gru_cell_9/mul_2Mulgru_cell_9/sub:z:0gru_cell_9/Relu:activations:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/mul_2�
gru_cell_9/add_3AddV2gru_cell_9/mul_1:z:0gru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/add_3�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����K   2
TensorArrayV2_1/element_shape�
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
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_9_readvariableop_resource)gru_cell_9_matmul_readvariableop_resource+gru_cell_9_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������K: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_27549106*
condR
while_cond_27549105*8
output_shapes'
%: : : : :���������K: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����K   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������K*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
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
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������K*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������K2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime�
IdentityIdentitystrided_slice_3:output:0!^gru_cell_9/MatMul/ReadVariableOp#^gru_cell_9/MatMul_1/ReadVariableOp^gru_cell_9/ReadVariableOp^while*
T0*'
_output_shapes
:���������K2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������:::2D
 gru_cell_9/MatMul/ReadVariableOp gru_cell_9/MatMul/ReadVariableOp2H
"gru_cell_9/MatMul_1/ReadVariableOp"gru_cell_9/MatMul_1/ReadVariableOp26
gru_cell_9/ReadVariableOpgru_cell_9/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
/__inference_sequential_9_layer_call_fn_27549933

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_sequential_9_layer_call_and_return_conditional_losses_275494842
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������:::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
$__inference__traced_restore_27551004
file_prefix$
 assignvariableop_dense_18_kernel$
 assignvariableop_1_dense_18_bias&
"assignvariableop_2_dense_19_kernel$
 assignvariableop_3_dense_19_bias 
assignvariableop_4_adam_iter"
assignvariableop_5_adam_beta_1"
assignvariableop_6_adam_beta_2!
assignvariableop_7_adam_decay)
%assignvariableop_8_adam_learning_rate.
*assignvariableop_9_gru_9_gru_cell_9_kernel9
5assignvariableop_10_gru_9_gru_cell_9_recurrent_kernel-
)assignvariableop_11_gru_9_gru_cell_9_bias
assignvariableop_12_total
assignvariableop_13_count
assignvariableop_14_total_1
assignvariableop_15_count_1
assignvariableop_16_total_2
assignvariableop_17_count_2.
*assignvariableop_18_adam_dense_18_kernel_m,
(assignvariableop_19_adam_dense_18_bias_m.
*assignvariableop_20_adam_dense_19_kernel_m,
(assignvariableop_21_adam_dense_19_bias_m6
2assignvariableop_22_adam_gru_9_gru_cell_9_kernel_m@
<assignvariableop_23_adam_gru_9_gru_cell_9_recurrent_kernel_m4
0assignvariableop_24_adam_gru_9_gru_cell_9_bias_m.
*assignvariableop_25_adam_dense_18_kernel_v,
(assignvariableop_26_adam_dense_18_bias_v.
*assignvariableop_27_adam_dense_19_kernel_v,
(assignvariableop_28_adam_dense_19_bias_v6
2assignvariableop_29_adam_gru_9_gru_cell_9_kernel_v@
<assignvariableop_30_adam_gru_9_gru_cell_9_recurrent_kernel_v4
0assignvariableop_31_adam_gru_9_gru_cell_9_bias_v
identity_33��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*�
value�B�!B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*U
valueLBJ!B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::*/
dtypes%
#2!	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp assignvariableop_dense_18_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_18_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_19_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_19_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp*assignvariableop_9_gru_9_gru_cell_9_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp5assignvariableop_10_gru_9_gru_cell_9_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp)assignvariableop_11_gru_9_gru_cell_9_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOpassignvariableop_12_totalIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOpassignvariableop_13_countIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOpassignvariableop_14_total_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOpassignvariableop_15_count_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOpassignvariableop_16_total_2Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOpassignvariableop_17_count_2Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp*assignvariableop_18_adam_dense_18_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp(assignvariableop_19_adam_dense_18_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_dense_19_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_dense_19_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp2assignvariableop_22_adam_gru_9_gru_cell_9_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp<assignvariableop_23_adam_gru_9_gru_cell_9_recurrent_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp0assignvariableop_24_adam_gru_9_gru_cell_9_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_18_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_18_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_dense_19_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_dense_19_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp2assignvariableop_29_adam_gru_9_gru_cell_9_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp<assignvariableop_30_adam_gru_9_gru_cell_9_recurrent_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp0assignvariableop_31_adam_gru_9_gru_cell_9_bias_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_319
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_32Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_32�
Identity_33IdentityIdentity_32:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_33"#
identity_33Identity_33:output:0*�
_input_shapes�
�: ::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_31AssignVariableOp_312(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
J__inference_sequential_9_layer_call_and_return_conditional_losses_27549439
gru_9_input
gru_9_27549378
gru_9_27549380
gru_9_27549382
dense_18_27549407
dense_18_27549409
dense_19_27549433
dense_19_27549435
identity�� dense_18/StatefulPartitionedCall� dense_19/StatefulPartitionedCall�gru_9/StatefulPartitionedCall�
gru_9/StatefulPartitionedCallStatefulPartitionedCallgru_9_inputgru_9_27549378gru_9_27549380gru_9_27549382*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������K*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_gru_9_layer_call_and_return_conditional_losses_275491962
gru_9/StatefulPartitionedCall�
 dense_18/StatefulPartitionedCallStatefulPartitionedCall&gru_9/StatefulPartitionedCall:output:0dense_18_27549407dense_18_27549409*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_18_layer_call_and_return_conditional_losses_275493962"
 dense_18/StatefulPartitionedCall�
 dense_19/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0dense_19_27549433dense_19_27549435*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_19_layer_call_and_return_conditional_losses_275494222"
 dense_19/StatefulPartitionedCall�
IdentityIdentity)dense_19/StatefulPartitionedCall:output:0!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall^gru_9/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������:::::::2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2>
gru_9/StatefulPartitionedCallgru_9/StatefulPartitionedCall:X T
+
_output_shapes
:���������
%
_user_specified_namegru_9_input
�
�
J__inference_sequential_9_layer_call_and_return_conditional_losses_27549460
gru_9_input
gru_9_27549442
gru_9_27549444
gru_9_27549446
dense_18_27549449
dense_18_27549451
dense_19_27549454
dense_19_27549456
identity�� dense_18/StatefulPartitionedCall� dense_19/StatefulPartitionedCall�gru_9/StatefulPartitionedCall�
gru_9/StatefulPartitionedCallStatefulPartitionedCallgru_9_inputgru_9_27549442gru_9_27549444gru_9_27549446*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������K*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_gru_9_layer_call_and_return_conditional_losses_275493552
gru_9/StatefulPartitionedCall�
 dense_18/StatefulPartitionedCallStatefulPartitionedCall&gru_9/StatefulPartitionedCall:output:0dense_18_27549449dense_18_27549451*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_18_layer_call_and_return_conditional_losses_275493962"
 dense_18/StatefulPartitionedCall�
 dense_19/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0dense_19_27549454dense_19_27549456*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_19_layer_call_and_return_conditional_losses_275494222"
 dense_19/StatefulPartitionedCall�
IdentityIdentity)dense_19/StatefulPartitionedCall:output:0!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall^gru_9/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������:::::::2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2>
gru_9/StatefulPartitionedCallgru_9/StatefulPartitionedCall:X T
+
_output_shapes
:���������
%
_user_specified_namegru_9_input
�F
�
while_body_27550021
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
1while_gru_cell_9_matmul_1_readvariableop_resource��&while/gru_cell_9/MatMul/ReadVariableOp�(while/gru_cell_9/MatMul_1/ReadVariableOp�while/gru_cell_9/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
while/gru_cell_9/ReadVariableOpReadVariableOp*while_gru_cell_9_readvariableop_resource_0*
_output_shapes
:	�*
dtype02!
while/gru_cell_9/ReadVariableOp�
while/gru_cell_9/unstackUnpack'while/gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2
while/gru_cell_9/unstack�
&while/gru_cell_9/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02(
&while/gru_cell_9/MatMul/ReadVariableOp�
while/gru_cell_9/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/gru_cell_9/MatMul�
while/gru_cell_9/BiasAddBiasAdd!while/gru_cell_9/MatMul:product:0!while/gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:����������2
while/gru_cell_9/BiasAddr
while/gru_cell_9/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_9/Const�
 while/gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2"
 while/gru_cell_9/split/split_dim�
while/gru_cell_9/splitSplit)while/gru_cell_9/split/split_dim:output:0!while/gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
while/gru_cell_9/split�
(while/gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes
:	K�*
dtype02*
(while/gru_cell_9/MatMul_1/ReadVariableOp�
while/gru_cell_9/MatMul_1MatMulwhile_placeholder_20while/gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/gru_cell_9/MatMul_1�
while/gru_cell_9/BiasAdd_1BiasAdd#while/gru_cell_9/MatMul_1:product:0!while/gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:����������2
while/gru_cell_9/BiasAdd_1�
while/gru_cell_9/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"K   K   ����2
while/gru_cell_9/Const_1�
"while/gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2$
"while/gru_cell_9/split_1/split_dim�
while/gru_cell_9/split_1SplitV#while/gru_cell_9/BiasAdd_1:output:0!while/gru_cell_9/Const_1:output:0+while/gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
while/gru_cell_9/split_1�
while/gru_cell_9/addAddV2while/gru_cell_9/split:output:0!while/gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/add�
while/gru_cell_9/SigmoidSigmoidwhile/gru_cell_9/add:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/Sigmoid�
while/gru_cell_9/add_1AddV2while/gru_cell_9/split:output:1!while/gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/add_1�
while/gru_cell_9/Sigmoid_1Sigmoidwhile/gru_cell_9/add_1:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/Sigmoid_1�
while/gru_cell_9/mulMulwhile/gru_cell_9/Sigmoid_1:y:0!while/gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/mul�
while/gru_cell_9/add_2AddV2while/gru_cell_9/split:output:2while/gru_cell_9/mul:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/add_2�
while/gru_cell_9/ReluReluwhile/gru_cell_9/add_2:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/Relu�
while/gru_cell_9/mul_1Mulwhile/gru_cell_9/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/mul_1u
while/gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
while/gru_cell_9/sub/x�
while/gru_cell_9/subSubwhile/gru_cell_9/sub/x:output:0while/gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/sub�
while/gru_cell_9/mul_2Mulwhile/gru_cell_9/sub:z:0#while/gru_cell_9/Relu:activations:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/mul_2�
while/gru_cell_9/add_3AddV2while/gru_cell_9/mul_1:z:0while/gru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/add_3�
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
while/add_1�
while/IdentityIdentitywhile/add_1:z:0'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/gru_cell_9/add_3:z:0'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*'
_output_shapes
:���������K2
while/Identity_4"h
1while_gru_cell_9_matmul_1_readvariableop_resource3while_gru_cell_9_matmul_1_readvariableop_resource_0"d
/while_gru_cell_9_matmul_readvariableop_resource1while_gru_cell_9_matmul_readvariableop_resource_0"V
(while_gru_cell_9_readvariableop_resource*while_gru_cell_9_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :���������K: : :::2P
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
:���������K:

_output_shapes
: :

_output_shapes
: 
�
�
+__inference_dense_19_layer_call_fn_27550671

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_19_layer_call_and_return_conditional_losses_275494222
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������d::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
(__inference_gru_9_layer_call_fn_27550281
inputs_0
unknown
	unknown_0
	unknown_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������K*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_gru_9_layer_call_and_return_conditional_losses_275489072
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������K2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�
�
+__inference_dense_18_layer_call_fn_27550652

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_18_layer_call_and_return_conditional_losses_275493962
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������K::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������K
 
_user_specified_nameinputs
�e
�
&sequential_9_gru_9_while_body_27548369B
>sequential_9_gru_9_while_sequential_9_gru_9_while_loop_counterH
Dsequential_9_gru_9_while_sequential_9_gru_9_while_maximum_iterations(
$sequential_9_gru_9_while_placeholder*
&sequential_9_gru_9_while_placeholder_1*
&sequential_9_gru_9_while_placeholder_2A
=sequential_9_gru_9_while_sequential_9_gru_9_strided_slice_1_0}
ysequential_9_gru_9_while_tensorarrayv2read_tensorlistgetitem_sequential_9_gru_9_tensorarrayunstack_tensorlistfromtensor_0A
=sequential_9_gru_9_while_gru_cell_9_readvariableop_resource_0H
Dsequential_9_gru_9_while_gru_cell_9_matmul_readvariableop_resource_0J
Fsequential_9_gru_9_while_gru_cell_9_matmul_1_readvariableop_resource_0%
!sequential_9_gru_9_while_identity'
#sequential_9_gru_9_while_identity_1'
#sequential_9_gru_9_while_identity_2'
#sequential_9_gru_9_while_identity_3'
#sequential_9_gru_9_while_identity_4?
;sequential_9_gru_9_while_sequential_9_gru_9_strided_slice_1{
wsequential_9_gru_9_while_tensorarrayv2read_tensorlistgetitem_sequential_9_gru_9_tensorarrayunstack_tensorlistfromtensor?
;sequential_9_gru_9_while_gru_cell_9_readvariableop_resourceF
Bsequential_9_gru_9_while_gru_cell_9_matmul_readvariableop_resourceH
Dsequential_9_gru_9_while_gru_cell_9_matmul_1_readvariableop_resource��9sequential_9/gru_9/while/gru_cell_9/MatMul/ReadVariableOp�;sequential_9/gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp�2sequential_9/gru_9/while/gru_cell_9/ReadVariableOp�
Jsequential_9/gru_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2L
Jsequential_9/gru_9/while/TensorArrayV2Read/TensorListGetItem/element_shape�
<sequential_9/gru_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemysequential_9_gru_9_while_tensorarrayv2read_tensorlistgetitem_sequential_9_gru_9_tensorarrayunstack_tensorlistfromtensor_0$sequential_9_gru_9_while_placeholderSsequential_9/gru_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02>
<sequential_9/gru_9/while/TensorArrayV2Read/TensorListGetItem�
2sequential_9/gru_9/while/gru_cell_9/ReadVariableOpReadVariableOp=sequential_9_gru_9_while_gru_cell_9_readvariableop_resource_0*
_output_shapes
:	�*
dtype024
2sequential_9/gru_9/while/gru_cell_9/ReadVariableOp�
+sequential_9/gru_9/while/gru_cell_9/unstackUnpack:sequential_9/gru_9/while/gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2-
+sequential_9/gru_9/while/gru_cell_9/unstack�
9sequential_9/gru_9/while/gru_cell_9/MatMul/ReadVariableOpReadVariableOpDsequential_9_gru_9_while_gru_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02;
9sequential_9/gru_9/while/gru_cell_9/MatMul/ReadVariableOp�
*sequential_9/gru_9/while/gru_cell_9/MatMulMatMulCsequential_9/gru_9/while/TensorArrayV2Read/TensorListGetItem:item:0Asequential_9/gru_9/while/gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2,
*sequential_9/gru_9/while/gru_cell_9/MatMul�
+sequential_9/gru_9/while/gru_cell_9/BiasAddBiasAdd4sequential_9/gru_9/while/gru_cell_9/MatMul:product:04sequential_9/gru_9/while/gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:����������2-
+sequential_9/gru_9/while/gru_cell_9/BiasAdd�
)sequential_9/gru_9/while/gru_cell_9/ConstConst*
_output_shapes
: *
dtype0*
value	B :2+
)sequential_9/gru_9/while/gru_cell_9/Const�
3sequential_9/gru_9/while/gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������25
3sequential_9/gru_9/while/gru_cell_9/split/split_dim�
)sequential_9/gru_9/while/gru_cell_9/splitSplit<sequential_9/gru_9/while/gru_cell_9/split/split_dim:output:04sequential_9/gru_9/while/gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2+
)sequential_9/gru_9/while/gru_cell_9/split�
;sequential_9/gru_9/while/gru_cell_9/MatMul_1/ReadVariableOpReadVariableOpFsequential_9_gru_9_while_gru_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes
:	K�*
dtype02=
;sequential_9/gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp�
,sequential_9/gru_9/while/gru_cell_9/MatMul_1MatMul&sequential_9_gru_9_while_placeholder_2Csequential_9/gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2.
,sequential_9/gru_9/while/gru_cell_9/MatMul_1�
-sequential_9/gru_9/while/gru_cell_9/BiasAdd_1BiasAdd6sequential_9/gru_9/while/gru_cell_9/MatMul_1:product:04sequential_9/gru_9/while/gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:����������2/
-sequential_9/gru_9/while/gru_cell_9/BiasAdd_1�
+sequential_9/gru_9/while/gru_cell_9/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"K   K   ����2-
+sequential_9/gru_9/while/gru_cell_9/Const_1�
5sequential_9/gru_9/while/gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������27
5sequential_9/gru_9/while/gru_cell_9/split_1/split_dim�
+sequential_9/gru_9/while/gru_cell_9/split_1SplitV6sequential_9/gru_9/while/gru_cell_9/BiasAdd_1:output:04sequential_9/gru_9/while/gru_cell_9/Const_1:output:0>sequential_9/gru_9/while/gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2-
+sequential_9/gru_9/while/gru_cell_9/split_1�
'sequential_9/gru_9/while/gru_cell_9/addAddV22sequential_9/gru_9/while/gru_cell_9/split:output:04sequential_9/gru_9/while/gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:���������K2)
'sequential_9/gru_9/while/gru_cell_9/add�
+sequential_9/gru_9/while/gru_cell_9/SigmoidSigmoid+sequential_9/gru_9/while/gru_cell_9/add:z:0*
T0*'
_output_shapes
:���������K2-
+sequential_9/gru_9/while/gru_cell_9/Sigmoid�
)sequential_9/gru_9/while/gru_cell_9/add_1AddV22sequential_9/gru_9/while/gru_cell_9/split:output:14sequential_9/gru_9/while/gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:���������K2+
)sequential_9/gru_9/while/gru_cell_9/add_1�
-sequential_9/gru_9/while/gru_cell_9/Sigmoid_1Sigmoid-sequential_9/gru_9/while/gru_cell_9/add_1:z:0*
T0*'
_output_shapes
:���������K2/
-sequential_9/gru_9/while/gru_cell_9/Sigmoid_1�
'sequential_9/gru_9/while/gru_cell_9/mulMul1sequential_9/gru_9/while/gru_cell_9/Sigmoid_1:y:04sequential_9/gru_9/while/gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:���������K2)
'sequential_9/gru_9/while/gru_cell_9/mul�
)sequential_9/gru_9/while/gru_cell_9/add_2AddV22sequential_9/gru_9/while/gru_cell_9/split:output:2+sequential_9/gru_9/while/gru_cell_9/mul:z:0*
T0*'
_output_shapes
:���������K2+
)sequential_9/gru_9/while/gru_cell_9/add_2�
(sequential_9/gru_9/while/gru_cell_9/ReluRelu-sequential_9/gru_9/while/gru_cell_9/add_2:z:0*
T0*'
_output_shapes
:���������K2*
(sequential_9/gru_9/while/gru_cell_9/Relu�
)sequential_9/gru_9/while/gru_cell_9/mul_1Mul/sequential_9/gru_9/while/gru_cell_9/Sigmoid:y:0&sequential_9_gru_9_while_placeholder_2*
T0*'
_output_shapes
:���������K2+
)sequential_9/gru_9/while/gru_cell_9/mul_1�
)sequential_9/gru_9/while/gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2+
)sequential_9/gru_9/while/gru_cell_9/sub/x�
'sequential_9/gru_9/while/gru_cell_9/subSub2sequential_9/gru_9/while/gru_cell_9/sub/x:output:0/sequential_9/gru_9/while/gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:���������K2)
'sequential_9/gru_9/while/gru_cell_9/sub�
)sequential_9/gru_9/while/gru_cell_9/mul_2Mul+sequential_9/gru_9/while/gru_cell_9/sub:z:06sequential_9/gru_9/while/gru_cell_9/Relu:activations:0*
T0*'
_output_shapes
:���������K2+
)sequential_9/gru_9/while/gru_cell_9/mul_2�
)sequential_9/gru_9/while/gru_cell_9/add_3AddV2-sequential_9/gru_9/while/gru_cell_9/mul_1:z:0-sequential_9/gru_9/while/gru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:���������K2+
)sequential_9/gru_9/while/gru_cell_9/add_3�
=sequential_9/gru_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem&sequential_9_gru_9_while_placeholder_1$sequential_9_gru_9_while_placeholder-sequential_9/gru_9/while/gru_cell_9/add_3:z:0*
_output_shapes
: *
element_dtype02?
=sequential_9/gru_9/while/TensorArrayV2Write/TensorListSetItem�
sequential_9/gru_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sequential_9/gru_9/while/add/y�
sequential_9/gru_9/while/addAddV2$sequential_9_gru_9_while_placeholder'sequential_9/gru_9/while/add/y:output:0*
T0*
_output_shapes
: 2
sequential_9/gru_9/while/add�
 sequential_9/gru_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_9/gru_9/while/add_1/y�
sequential_9/gru_9/while/add_1AddV2>sequential_9_gru_9_while_sequential_9_gru_9_while_loop_counter)sequential_9/gru_9/while/add_1/y:output:0*
T0*
_output_shapes
: 2 
sequential_9/gru_9/while/add_1�
!sequential_9/gru_9/while/IdentityIdentity"sequential_9/gru_9/while/add_1:z:0:^sequential_9/gru_9/while/gru_cell_9/MatMul/ReadVariableOp<^sequential_9/gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp3^sequential_9/gru_9/while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2#
!sequential_9/gru_9/while/Identity�
#sequential_9/gru_9/while/Identity_1IdentityDsequential_9_gru_9_while_sequential_9_gru_9_while_maximum_iterations:^sequential_9/gru_9/while/gru_cell_9/MatMul/ReadVariableOp<^sequential_9/gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp3^sequential_9/gru_9/while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2%
#sequential_9/gru_9/while/Identity_1�
#sequential_9/gru_9/while/Identity_2Identity sequential_9/gru_9/while/add:z:0:^sequential_9/gru_9/while/gru_cell_9/MatMul/ReadVariableOp<^sequential_9/gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp3^sequential_9/gru_9/while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2%
#sequential_9/gru_9/while/Identity_2�
#sequential_9/gru_9/while/Identity_3IdentityMsequential_9/gru_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0:^sequential_9/gru_9/while/gru_cell_9/MatMul/ReadVariableOp<^sequential_9/gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp3^sequential_9/gru_9/while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2%
#sequential_9/gru_9/while/Identity_3�
#sequential_9/gru_9/while/Identity_4Identity-sequential_9/gru_9/while/gru_cell_9/add_3:z:0:^sequential_9/gru_9/while/gru_cell_9/MatMul/ReadVariableOp<^sequential_9/gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp3^sequential_9/gru_9/while/gru_cell_9/ReadVariableOp*
T0*'
_output_shapes
:���������K2%
#sequential_9/gru_9/while/Identity_4"�
Dsequential_9_gru_9_while_gru_cell_9_matmul_1_readvariableop_resourceFsequential_9_gru_9_while_gru_cell_9_matmul_1_readvariableop_resource_0"�
Bsequential_9_gru_9_while_gru_cell_9_matmul_readvariableop_resourceDsequential_9_gru_9_while_gru_cell_9_matmul_readvariableop_resource_0"|
;sequential_9_gru_9_while_gru_cell_9_readvariableop_resource=sequential_9_gru_9_while_gru_cell_9_readvariableop_resource_0"O
!sequential_9_gru_9_while_identity*sequential_9/gru_9/while/Identity:output:0"S
#sequential_9_gru_9_while_identity_1,sequential_9/gru_9/while/Identity_1:output:0"S
#sequential_9_gru_9_while_identity_2,sequential_9/gru_9/while/Identity_2:output:0"S
#sequential_9_gru_9_while_identity_3,sequential_9/gru_9/while/Identity_3:output:0"S
#sequential_9_gru_9_while_identity_4,sequential_9/gru_9/while/Identity_4:output:0"|
;sequential_9_gru_9_while_sequential_9_gru_9_strided_slice_1=sequential_9_gru_9_while_sequential_9_gru_9_strided_slice_1_0"�
wsequential_9_gru_9_while_tensorarrayv2read_tensorlistgetitem_sequential_9_gru_9_tensorarrayunstack_tensorlistfromtensorysequential_9_gru_9_while_tensorarrayv2read_tensorlistgetitem_sequential_9_gru_9_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :���������K: : :::2v
9sequential_9/gru_9/while/gru_cell_9/MatMul/ReadVariableOp9sequential_9/gru_9/while/gru_cell_9/MatMul/ReadVariableOp2z
;sequential_9/gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp;sequential_9/gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp2h
2sequential_9/gru_9/while/gru_cell_9/ReadVariableOp2sequential_9/gru_9/while/gru_cell_9/ReadVariableOp: 
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
:���������K:

_output_shapes
: :

_output_shapes
: 
�
�
/__inference_sequential_9_layer_call_fn_27549541
gru_9_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallgru_9_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_sequential_9_layer_call_and_return_conditional_losses_275495242
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������:::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:���������
%
_user_specified_namegru_9_input
�
�
H__inference_gru_cell_9_layer_call_and_return_conditional_losses_27550711

inputs
states_0
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	�*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2	
unstack�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:����������2	
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
���������2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
split�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	K�*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:����������2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"K   K   ����2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split_1/split_dim�
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:���������K2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������K2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:���������K2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������K2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:���������K2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:���������K2
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:���������K2
Relu^
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:���������K2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������K2
subd
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:���������K2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:���������K2
add_3�
IdentityIdentity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:���������K2

Identity�

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:���������K2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:���������:���������K:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������K
"
_user_specified_name
states/0
�	
�
F__inference_dense_18_layer_call_and_return_conditional_losses_27550643

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Kd*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������d2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������K::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������K
 
_user_specified_nameinputs
�<
�
C__inference_gru_9_layer_call_and_return_conditional_losses_27548907

inputs
gru_cell_9_27548831
gru_cell_9_27548833
gru_cell_9_27548835
identity��"gru_cell_9/StatefulPartitionedCall�whileD
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
strided_slice/stack_2�
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
B :�2
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
zeros/packed/1�
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
:���������K2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
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
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
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
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
"gru_cell_9/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_9_27548831gru_cell_9_27548833gru_cell_9_27548835*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������K:���������K*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_gru_cell_9_layer_call_and_return_conditional_losses_275485442$
"gru_cell_9/StatefulPartitionedCall�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����K   2
TensorArrayV2_1/element_shape�
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
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_9_27548831gru_cell_9_27548833gru_cell_9_27548835*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������K: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_27548843*
condR
while_cond_27548842*8
output_shapes'
%: : : : :���������K: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����K   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������K*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
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
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������K*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������K2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime�
IdentityIdentitystrided_slice_3:output:0#^gru_cell_9/StatefulPartitionedCall^while*
T0*'
_output_shapes
:���������K2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::2H
"gru_cell_9/StatefulPartitionedCall"gru_cell_9/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�F
�
while_body_27549106
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
1while_gru_cell_9_matmul_1_readvariableop_resource��&while/gru_cell_9/MatMul/ReadVariableOp�(while/gru_cell_9/MatMul_1/ReadVariableOp�while/gru_cell_9/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
while/gru_cell_9/ReadVariableOpReadVariableOp*while_gru_cell_9_readvariableop_resource_0*
_output_shapes
:	�*
dtype02!
while/gru_cell_9/ReadVariableOp�
while/gru_cell_9/unstackUnpack'while/gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2
while/gru_cell_9/unstack�
&while/gru_cell_9/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02(
&while/gru_cell_9/MatMul/ReadVariableOp�
while/gru_cell_9/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/gru_cell_9/MatMul�
while/gru_cell_9/BiasAddBiasAdd!while/gru_cell_9/MatMul:product:0!while/gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:����������2
while/gru_cell_9/BiasAddr
while/gru_cell_9/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_9/Const�
 while/gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2"
 while/gru_cell_9/split/split_dim�
while/gru_cell_9/splitSplit)while/gru_cell_9/split/split_dim:output:0!while/gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
while/gru_cell_9/split�
(while/gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes
:	K�*
dtype02*
(while/gru_cell_9/MatMul_1/ReadVariableOp�
while/gru_cell_9/MatMul_1MatMulwhile_placeholder_20while/gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/gru_cell_9/MatMul_1�
while/gru_cell_9/BiasAdd_1BiasAdd#while/gru_cell_9/MatMul_1:product:0!while/gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:����������2
while/gru_cell_9/BiasAdd_1�
while/gru_cell_9/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"K   K   ����2
while/gru_cell_9/Const_1�
"while/gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2$
"while/gru_cell_9/split_1/split_dim�
while/gru_cell_9/split_1SplitV#while/gru_cell_9/BiasAdd_1:output:0!while/gru_cell_9/Const_1:output:0+while/gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
while/gru_cell_9/split_1�
while/gru_cell_9/addAddV2while/gru_cell_9/split:output:0!while/gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/add�
while/gru_cell_9/SigmoidSigmoidwhile/gru_cell_9/add:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/Sigmoid�
while/gru_cell_9/add_1AddV2while/gru_cell_9/split:output:1!while/gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/add_1�
while/gru_cell_9/Sigmoid_1Sigmoidwhile/gru_cell_9/add_1:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/Sigmoid_1�
while/gru_cell_9/mulMulwhile/gru_cell_9/Sigmoid_1:y:0!while/gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/mul�
while/gru_cell_9/add_2AddV2while/gru_cell_9/split:output:2while/gru_cell_9/mul:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/add_2�
while/gru_cell_9/ReluReluwhile/gru_cell_9/add_2:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/Relu�
while/gru_cell_9/mul_1Mulwhile/gru_cell_9/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/mul_1u
while/gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
while/gru_cell_9/sub/x�
while/gru_cell_9/subSubwhile/gru_cell_9/sub/x:output:0while/gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/sub�
while/gru_cell_9/mul_2Mulwhile/gru_cell_9/sub:z:0#while/gru_cell_9/Relu:activations:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/mul_2�
while/gru_cell_9/add_3AddV2while/gru_cell_9/mul_1:z:0while/gru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/add_3�
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
while/add_1�
while/IdentityIdentitywhile/add_1:z:0'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/gru_cell_9/add_3:z:0'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*'
_output_shapes
:���������K2
while/Identity_4"h
1while_gru_cell_9_matmul_1_readvariableop_resource3while_gru_cell_9_matmul_1_readvariableop_resource_0"d
/while_gru_cell_9_matmul_readvariableop_resource1while_gru_cell_9_matmul_readvariableop_resource_0"V
(while_gru_cell_9_readvariableop_resource*while_gru_cell_9_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :���������K: : :::2P
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
:���������K:

_output_shapes
: :

_output_shapes
: 
�
�
(__inference_gru_9_layer_call_fn_27550632

inputs
unknown
	unknown_0
	unknown_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������K*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_gru_9_layer_call_and_return_conditional_losses_275493552
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������K2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
-__inference_gru_cell_9_layer_call_fn_27550779

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������K:���������K*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_gru_cell_9_layer_call_and_return_conditional_losses_275485842
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������K2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������K2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:���������:���������K:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������K
"
_user_specified_name
states/0
�Z
�
C__inference_gru_9_layer_call_and_return_conditional_losses_27550111
inputs_0&
"gru_cell_9_readvariableop_resource-
)gru_cell_9_matmul_readvariableop_resource/
+gru_cell_9_matmul_1_readvariableop_resource
identity�� gru_cell_9/MatMul/ReadVariableOp�"gru_cell_9/MatMul_1/ReadVariableOp�gru_cell_9/ReadVariableOp�whileF
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
strided_slice/stack_2�
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
B :�2
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
zeros/packed/1�
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
:���������K2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
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
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
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
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
gru_cell_9/ReadVariableOpReadVariableOp"gru_cell_9_readvariableop_resource*
_output_shapes
:	�*
dtype02
gru_cell_9/ReadVariableOp�
gru_cell_9/unstackUnpack!gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2
gru_cell_9/unstack�
 gru_cell_9/MatMul/ReadVariableOpReadVariableOp)gru_cell_9_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02"
 gru_cell_9/MatMul/ReadVariableOp�
gru_cell_9/MatMulMatMulstrided_slice_2:output:0(gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
gru_cell_9/MatMul�
gru_cell_9/BiasAddBiasAddgru_cell_9/MatMul:product:0gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:����������2
gru_cell_9/BiasAddf
gru_cell_9/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_9/Const�
gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
gru_cell_9/split/split_dim�
gru_cell_9/splitSplit#gru_cell_9/split/split_dim:output:0gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
gru_cell_9/split�
"gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_9_matmul_1_readvariableop_resource*
_output_shapes
:	K�*
dtype02$
"gru_cell_9/MatMul_1/ReadVariableOp�
gru_cell_9/MatMul_1MatMulzeros:output:0*gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
gru_cell_9/MatMul_1�
gru_cell_9/BiasAdd_1BiasAddgru_cell_9/MatMul_1:product:0gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:����������2
gru_cell_9/BiasAdd_1}
gru_cell_9/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"K   K   ����2
gru_cell_9/Const_1�
gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
gru_cell_9/split_1/split_dim�
gru_cell_9/split_1SplitVgru_cell_9/BiasAdd_1:output:0gru_cell_9/Const_1:output:0%gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
gru_cell_9/split_1�
gru_cell_9/addAddV2gru_cell_9/split:output:0gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/addy
gru_cell_9/SigmoidSigmoidgru_cell_9/add:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/Sigmoid�
gru_cell_9/add_1AddV2gru_cell_9/split:output:1gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:���������K2
gru_cell_9/add_1
gru_cell_9/Sigmoid_1Sigmoidgru_cell_9/add_1:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/Sigmoid_1�
gru_cell_9/mulMulgru_cell_9/Sigmoid_1:y:0gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:���������K2
gru_cell_9/mul�
gru_cell_9/add_2AddV2gru_cell_9/split:output:2gru_cell_9/mul:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/add_2r
gru_cell_9/ReluRelugru_cell_9/add_2:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/Relu�
gru_cell_9/mul_1Mulgru_cell_9/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/mul_1i
gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
gru_cell_9/sub/x�
gru_cell_9/subSubgru_cell_9/sub/x:output:0gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/sub�
gru_cell_9/mul_2Mulgru_cell_9/sub:z:0gru_cell_9/Relu:activations:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/mul_2�
gru_cell_9/add_3AddV2gru_cell_9/mul_1:z:0gru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/add_3�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����K   2
TensorArrayV2_1/element_shape�
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
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_9_readvariableop_resource)gru_cell_9_matmul_readvariableop_resource+gru_cell_9_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������K: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_27550021*
condR
while_cond_27550020*8
output_shapes'
%: : : : :���������K: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����K   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������K*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
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
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������K*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������K2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime�
IdentityIdentitystrided_slice_3:output:0!^gru_cell_9/MatMul/ReadVariableOp#^gru_cell_9/MatMul_1/ReadVariableOp^gru_cell_9/ReadVariableOp^while*
T0*'
_output_shapes
:���������K2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::2D
 gru_cell_9/MatMul/ReadVariableOp gru_cell_9/MatMul/ReadVariableOp2H
"gru_cell_9/MatMul_1/ReadVariableOp"gru_cell_9/MatMul_1/ReadVariableOp26
gru_cell_9/ReadVariableOpgru_cell_9/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�
�
while_cond_27550519
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_27550519___redundant_placeholder06
2while_while_cond_27550519___redundant_placeholder16
2while_while_cond_27550519___redundant_placeholder26
2while_while_cond_27550519___redundant_placeholder3
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
-: : : : :���������K: ::::: 
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
:���������K:

_output_shapes
: :

_output_shapes
:
�P
�
gru_9_while_body_27549811(
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
7gru_9_while_gru_cell_9_matmul_1_readvariableop_resource��,gru_9/while/gru_cell_9/MatMul/ReadVariableOp�.gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp�%gru_9/while/gru_cell_9/ReadVariableOp�
=gru_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2?
=gru_9/while/TensorArrayV2Read/TensorListGetItem/element_shape�
/gru_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_9_while_tensorarrayv2read_tensorlistgetitem_gru_9_tensorarrayunstack_tensorlistfromtensor_0gru_9_while_placeholderFgru_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype021
/gru_9/while/TensorArrayV2Read/TensorListGetItem�
%gru_9/while/gru_cell_9/ReadVariableOpReadVariableOp0gru_9_while_gru_cell_9_readvariableop_resource_0*
_output_shapes
:	�*
dtype02'
%gru_9/while/gru_cell_9/ReadVariableOp�
gru_9/while/gru_cell_9/unstackUnpack-gru_9/while/gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2 
gru_9/while/gru_cell_9/unstack�
,gru_9/while/gru_cell_9/MatMul/ReadVariableOpReadVariableOp7gru_9_while_gru_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02.
,gru_9/while/gru_cell_9/MatMul/ReadVariableOp�
gru_9/while/gru_cell_9/MatMulMatMul6gru_9/while/TensorArrayV2Read/TensorListGetItem:item:04gru_9/while/gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
gru_9/while/gru_cell_9/MatMul�
gru_9/while/gru_cell_9/BiasAddBiasAdd'gru_9/while/gru_cell_9/MatMul:product:0'gru_9/while/gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:����������2 
gru_9/while/gru_cell_9/BiasAdd~
gru_9/while/gru_cell_9/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_9/while/gru_cell_9/Const�
&gru_9/while/gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2(
&gru_9/while/gru_cell_9/split/split_dim�
gru_9/while/gru_cell_9/splitSplit/gru_9/while/gru_cell_9/split/split_dim:output:0'gru_9/while/gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
gru_9/while/gru_cell_9/split�
.gru_9/while/gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp9gru_9_while_gru_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes
:	K�*
dtype020
.gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp�
gru_9/while/gru_cell_9/MatMul_1MatMulgru_9_while_placeholder_26gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2!
gru_9/while/gru_cell_9/MatMul_1�
 gru_9/while/gru_cell_9/BiasAdd_1BiasAdd)gru_9/while/gru_cell_9/MatMul_1:product:0'gru_9/while/gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:����������2"
 gru_9/while/gru_cell_9/BiasAdd_1�
gru_9/while/gru_cell_9/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"K   K   ����2 
gru_9/while/gru_cell_9/Const_1�
(gru_9/while/gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2*
(gru_9/while/gru_cell_9/split_1/split_dim�
gru_9/while/gru_cell_9/split_1SplitV)gru_9/while/gru_cell_9/BiasAdd_1:output:0'gru_9/while/gru_cell_9/Const_1:output:01gru_9/while/gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2 
gru_9/while/gru_cell_9/split_1�
gru_9/while/gru_cell_9/addAddV2%gru_9/while/gru_cell_9/split:output:0'gru_9/while/gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:���������K2
gru_9/while/gru_cell_9/add�
gru_9/while/gru_cell_9/SigmoidSigmoidgru_9/while/gru_cell_9/add:z:0*
T0*'
_output_shapes
:���������K2 
gru_9/while/gru_cell_9/Sigmoid�
gru_9/while/gru_cell_9/add_1AddV2%gru_9/while/gru_cell_9/split:output:1'gru_9/while/gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:���������K2
gru_9/while/gru_cell_9/add_1�
 gru_9/while/gru_cell_9/Sigmoid_1Sigmoid gru_9/while/gru_cell_9/add_1:z:0*
T0*'
_output_shapes
:���������K2"
 gru_9/while/gru_cell_9/Sigmoid_1�
gru_9/while/gru_cell_9/mulMul$gru_9/while/gru_cell_9/Sigmoid_1:y:0'gru_9/while/gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:���������K2
gru_9/while/gru_cell_9/mul�
gru_9/while/gru_cell_9/add_2AddV2%gru_9/while/gru_cell_9/split:output:2gru_9/while/gru_cell_9/mul:z:0*
T0*'
_output_shapes
:���������K2
gru_9/while/gru_cell_9/add_2�
gru_9/while/gru_cell_9/ReluRelu gru_9/while/gru_cell_9/add_2:z:0*
T0*'
_output_shapes
:���������K2
gru_9/while/gru_cell_9/Relu�
gru_9/while/gru_cell_9/mul_1Mul"gru_9/while/gru_cell_9/Sigmoid:y:0gru_9_while_placeholder_2*
T0*'
_output_shapes
:���������K2
gru_9/while/gru_cell_9/mul_1�
gru_9/while/gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
gru_9/while/gru_cell_9/sub/x�
gru_9/while/gru_cell_9/subSub%gru_9/while/gru_cell_9/sub/x:output:0"gru_9/while/gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:���������K2
gru_9/while/gru_cell_9/sub�
gru_9/while/gru_cell_9/mul_2Mulgru_9/while/gru_cell_9/sub:z:0)gru_9/while/gru_cell_9/Relu:activations:0*
T0*'
_output_shapes
:���������K2
gru_9/while/gru_cell_9/mul_2�
gru_9/while/gru_cell_9/add_3AddV2 gru_9/while/gru_cell_9/mul_1:z:0 gru_9/while/gru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:���������K2
gru_9/while/gru_cell_9/add_3�
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
gru_9/while/add/y�
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
gru_9/while/add_1/y�
gru_9/while/add_1AddV2$gru_9_while_gru_9_while_loop_countergru_9/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru_9/while/add_1�
gru_9/while/IdentityIdentitygru_9/while/add_1:z:0-^gru_9/while/gru_cell_9/MatMul/ReadVariableOp/^gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp&^gru_9/while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
gru_9/while/Identity�
gru_9/while/Identity_1Identity*gru_9_while_gru_9_while_maximum_iterations-^gru_9/while/gru_cell_9/MatMul/ReadVariableOp/^gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp&^gru_9/while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
gru_9/while/Identity_1�
gru_9/while/Identity_2Identitygru_9/while/add:z:0-^gru_9/while/gru_cell_9/MatMul/ReadVariableOp/^gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp&^gru_9/while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
gru_9/while/Identity_2�
gru_9/while/Identity_3Identity@gru_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0-^gru_9/while/gru_cell_9/MatMul/ReadVariableOp/^gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp&^gru_9/while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
gru_9/while/Identity_3�
gru_9/while/Identity_4Identity gru_9/while/gru_cell_9/add_3:z:0-^gru_9/while/gru_cell_9/MatMul/ReadVariableOp/^gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp&^gru_9/while/gru_cell_9/ReadVariableOp*
T0*'
_output_shapes
:���������K2
gru_9/while/Identity_4"H
!gru_9_while_gru_9_strided_slice_1#gru_9_while_gru_9_strided_slice_1_0"t
7gru_9_while_gru_cell_9_matmul_1_readvariableop_resource9gru_9_while_gru_cell_9_matmul_1_readvariableop_resource_0"p
5gru_9_while_gru_cell_9_matmul_readvariableop_resource7gru_9_while_gru_cell_9_matmul_readvariableop_resource_0"b
.gru_9_while_gru_cell_9_readvariableop_resource0gru_9_while_gru_cell_9_readvariableop_resource_0"5
gru_9_while_identitygru_9/while/Identity:output:0"9
gru_9_while_identity_1gru_9/while/Identity_1:output:0"9
gru_9_while_identity_2gru_9/while/Identity_2:output:0"9
gru_9_while_identity_3gru_9/while/Identity_3:output:0"9
gru_9_while_identity_4gru_9/while/Identity_4:output:0"�
]gru_9_while_tensorarrayv2read_tensorlistgetitem_gru_9_tensorarrayunstack_tensorlistfromtensor_gru_9_while_tensorarrayv2read_tensorlistgetitem_gru_9_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :���������K: : :::2\
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
:���������K:

_output_shapes
: :

_output_shapes
: 
�P
�
gru_9_while_body_27549639(
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
7gru_9_while_gru_cell_9_matmul_1_readvariableop_resource��,gru_9/while/gru_cell_9/MatMul/ReadVariableOp�.gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp�%gru_9/while/gru_cell_9/ReadVariableOp�
=gru_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2?
=gru_9/while/TensorArrayV2Read/TensorListGetItem/element_shape�
/gru_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_9_while_tensorarrayv2read_tensorlistgetitem_gru_9_tensorarrayunstack_tensorlistfromtensor_0gru_9_while_placeholderFgru_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype021
/gru_9/while/TensorArrayV2Read/TensorListGetItem�
%gru_9/while/gru_cell_9/ReadVariableOpReadVariableOp0gru_9_while_gru_cell_9_readvariableop_resource_0*
_output_shapes
:	�*
dtype02'
%gru_9/while/gru_cell_9/ReadVariableOp�
gru_9/while/gru_cell_9/unstackUnpack-gru_9/while/gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2 
gru_9/while/gru_cell_9/unstack�
,gru_9/while/gru_cell_9/MatMul/ReadVariableOpReadVariableOp7gru_9_while_gru_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02.
,gru_9/while/gru_cell_9/MatMul/ReadVariableOp�
gru_9/while/gru_cell_9/MatMulMatMul6gru_9/while/TensorArrayV2Read/TensorListGetItem:item:04gru_9/while/gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
gru_9/while/gru_cell_9/MatMul�
gru_9/while/gru_cell_9/BiasAddBiasAdd'gru_9/while/gru_cell_9/MatMul:product:0'gru_9/while/gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:����������2 
gru_9/while/gru_cell_9/BiasAdd~
gru_9/while/gru_cell_9/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_9/while/gru_cell_9/Const�
&gru_9/while/gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2(
&gru_9/while/gru_cell_9/split/split_dim�
gru_9/while/gru_cell_9/splitSplit/gru_9/while/gru_cell_9/split/split_dim:output:0'gru_9/while/gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
gru_9/while/gru_cell_9/split�
.gru_9/while/gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp9gru_9_while_gru_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes
:	K�*
dtype020
.gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp�
gru_9/while/gru_cell_9/MatMul_1MatMulgru_9_while_placeholder_26gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2!
gru_9/while/gru_cell_9/MatMul_1�
 gru_9/while/gru_cell_9/BiasAdd_1BiasAdd)gru_9/while/gru_cell_9/MatMul_1:product:0'gru_9/while/gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:����������2"
 gru_9/while/gru_cell_9/BiasAdd_1�
gru_9/while/gru_cell_9/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"K   K   ����2 
gru_9/while/gru_cell_9/Const_1�
(gru_9/while/gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2*
(gru_9/while/gru_cell_9/split_1/split_dim�
gru_9/while/gru_cell_9/split_1SplitV)gru_9/while/gru_cell_9/BiasAdd_1:output:0'gru_9/while/gru_cell_9/Const_1:output:01gru_9/while/gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2 
gru_9/while/gru_cell_9/split_1�
gru_9/while/gru_cell_9/addAddV2%gru_9/while/gru_cell_9/split:output:0'gru_9/while/gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:���������K2
gru_9/while/gru_cell_9/add�
gru_9/while/gru_cell_9/SigmoidSigmoidgru_9/while/gru_cell_9/add:z:0*
T0*'
_output_shapes
:���������K2 
gru_9/while/gru_cell_9/Sigmoid�
gru_9/while/gru_cell_9/add_1AddV2%gru_9/while/gru_cell_9/split:output:1'gru_9/while/gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:���������K2
gru_9/while/gru_cell_9/add_1�
 gru_9/while/gru_cell_9/Sigmoid_1Sigmoid gru_9/while/gru_cell_9/add_1:z:0*
T0*'
_output_shapes
:���������K2"
 gru_9/while/gru_cell_9/Sigmoid_1�
gru_9/while/gru_cell_9/mulMul$gru_9/while/gru_cell_9/Sigmoid_1:y:0'gru_9/while/gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:���������K2
gru_9/while/gru_cell_9/mul�
gru_9/while/gru_cell_9/add_2AddV2%gru_9/while/gru_cell_9/split:output:2gru_9/while/gru_cell_9/mul:z:0*
T0*'
_output_shapes
:���������K2
gru_9/while/gru_cell_9/add_2�
gru_9/while/gru_cell_9/ReluRelu gru_9/while/gru_cell_9/add_2:z:0*
T0*'
_output_shapes
:���������K2
gru_9/while/gru_cell_9/Relu�
gru_9/while/gru_cell_9/mul_1Mul"gru_9/while/gru_cell_9/Sigmoid:y:0gru_9_while_placeholder_2*
T0*'
_output_shapes
:���������K2
gru_9/while/gru_cell_9/mul_1�
gru_9/while/gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
gru_9/while/gru_cell_9/sub/x�
gru_9/while/gru_cell_9/subSub%gru_9/while/gru_cell_9/sub/x:output:0"gru_9/while/gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:���������K2
gru_9/while/gru_cell_9/sub�
gru_9/while/gru_cell_9/mul_2Mulgru_9/while/gru_cell_9/sub:z:0)gru_9/while/gru_cell_9/Relu:activations:0*
T0*'
_output_shapes
:���������K2
gru_9/while/gru_cell_9/mul_2�
gru_9/while/gru_cell_9/add_3AddV2 gru_9/while/gru_cell_9/mul_1:z:0 gru_9/while/gru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:���������K2
gru_9/while/gru_cell_9/add_3�
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
gru_9/while/add/y�
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
gru_9/while/add_1/y�
gru_9/while/add_1AddV2$gru_9_while_gru_9_while_loop_countergru_9/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru_9/while/add_1�
gru_9/while/IdentityIdentitygru_9/while/add_1:z:0-^gru_9/while/gru_cell_9/MatMul/ReadVariableOp/^gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp&^gru_9/while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
gru_9/while/Identity�
gru_9/while/Identity_1Identity*gru_9_while_gru_9_while_maximum_iterations-^gru_9/while/gru_cell_9/MatMul/ReadVariableOp/^gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp&^gru_9/while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
gru_9/while/Identity_1�
gru_9/while/Identity_2Identitygru_9/while/add:z:0-^gru_9/while/gru_cell_9/MatMul/ReadVariableOp/^gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp&^gru_9/while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
gru_9/while/Identity_2�
gru_9/while/Identity_3Identity@gru_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0-^gru_9/while/gru_cell_9/MatMul/ReadVariableOp/^gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp&^gru_9/while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
gru_9/while/Identity_3�
gru_9/while/Identity_4Identity gru_9/while/gru_cell_9/add_3:z:0-^gru_9/while/gru_cell_9/MatMul/ReadVariableOp/^gru_9/while/gru_cell_9/MatMul_1/ReadVariableOp&^gru_9/while/gru_cell_9/ReadVariableOp*
T0*'
_output_shapes
:���������K2
gru_9/while/Identity_4"H
!gru_9_while_gru_9_strided_slice_1#gru_9_while_gru_9_strided_slice_1_0"t
7gru_9_while_gru_cell_9_matmul_1_readvariableop_resource9gru_9_while_gru_cell_9_matmul_1_readvariableop_resource_0"p
5gru_9_while_gru_cell_9_matmul_readvariableop_resource7gru_9_while_gru_cell_9_matmul_readvariableop_resource_0"b
.gru_9_while_gru_cell_9_readvariableop_resource0gru_9_while_gru_cell_9_readvariableop_resource_0"5
gru_9_while_identitygru_9/while/Identity:output:0"9
gru_9_while_identity_1gru_9/while/Identity_1:output:0"9
gru_9_while_identity_2gru_9/while/Identity_2:output:0"9
gru_9_while_identity_3gru_9/while/Identity_3:output:0"9
gru_9_while_identity_4gru_9/while/Identity_4:output:0"�
]gru_9_while_tensorarrayv2read_tensorlistgetitem_gru_9_tensorarrayunstack_tensorlistfromtensor_gru_9_while_tensorarrayv2read_tensorlistgetitem_gru_9_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :���������K: : :::2\
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
:���������K:

_output_shapes
: :

_output_shapes
: 
�
�
J__inference_sequential_9_layer_call_and_return_conditional_losses_27549524

inputs
gru_9_27549506
gru_9_27549508
gru_9_27549510
dense_18_27549513
dense_18_27549515
dense_19_27549518
dense_19_27549520
identity�� dense_18/StatefulPartitionedCall� dense_19/StatefulPartitionedCall�gru_9/StatefulPartitionedCall�
gru_9/StatefulPartitionedCallStatefulPartitionedCallinputsgru_9_27549506gru_9_27549508gru_9_27549510*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������K*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_gru_9_layer_call_and_return_conditional_losses_275493552
gru_9/StatefulPartitionedCall�
 dense_18/StatefulPartitionedCallStatefulPartitionedCall&gru_9/StatefulPartitionedCall:output:0dense_18_27549513dense_18_27549515*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_18_layer_call_and_return_conditional_losses_275493962"
 dense_18/StatefulPartitionedCall�
 dense_19/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0dense_19_27549518dense_19_27549520*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_19_layer_call_and_return_conditional_losses_275494222"
 dense_19/StatefulPartitionedCall�
IdentityIdentity)dense_19/StatefulPartitionedCall:output:0!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall^gru_9/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������:::::::2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2>
gru_9/StatefulPartitionedCallgru_9/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�w
�
J__inference_sequential_9_layer_call_and_return_conditional_losses_27549742

inputs,
(gru_9_gru_cell_9_readvariableop_resource3
/gru_9_gru_cell_9_matmul_readvariableop_resource5
1gru_9_gru_cell_9_matmul_1_readvariableop_resource+
'dense_18_matmul_readvariableop_resource,
(dense_18_biasadd_readvariableop_resource+
'dense_19_matmul_readvariableop_resource,
(dense_19_biasadd_readvariableop_resource
identity��dense_18/BiasAdd/ReadVariableOp�dense_18/MatMul/ReadVariableOp�dense_19/BiasAdd/ReadVariableOp�dense_19/MatMul/ReadVariableOp�&gru_9/gru_cell_9/MatMul/ReadVariableOp�(gru_9/gru_cell_9/MatMul_1/ReadVariableOp�gru_9/gru_cell_9/ReadVariableOp�gru_9/whileP
gru_9/ShapeShapeinputs*
T0*
_output_shapes
:2
gru_9/Shape�
gru_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_9/strided_slice/stack�
gru_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_9/strided_slice/stack_1�
gru_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_9/strided_slice/stack_2�
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
value	B :K2
gru_9/zeros/mul/y�
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
B :�2
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
value	B :K2
gru_9/zeros/packed/1�
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
gru_9/zeros/Const�
gru_9/zerosFillgru_9/zeros/packed:output:0gru_9/zeros/Const:output:0*
T0*'
_output_shapes
:���������K2
gru_9/zeros�
gru_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_9/transpose/perm�
gru_9/transpose	Transposeinputsgru_9/transpose/perm:output:0*
T0*+
_output_shapes
:���������2
gru_9/transposea
gru_9/Shape_1Shapegru_9/transpose:y:0*
T0*
_output_shapes
:2
gru_9/Shape_1�
gru_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_9/strided_slice_1/stack�
gru_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_9/strided_slice_1/stack_1�
gru_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_9/strided_slice_1/stack_2�
gru_9/strided_slice_1StridedSlicegru_9/Shape_1:output:0$gru_9/strided_slice_1/stack:output:0&gru_9/strided_slice_1/stack_1:output:0&gru_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_9/strided_slice_1�
!gru_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2#
!gru_9/TensorArrayV2/element_shape�
gru_9/TensorArrayV2TensorListReserve*gru_9/TensorArrayV2/element_shape:output:0gru_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_9/TensorArrayV2�
;gru_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2=
;gru_9/TensorArrayUnstack/TensorListFromTensor/element_shape�
-gru_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_9/transpose:y:0Dgru_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02/
-gru_9/TensorArrayUnstack/TensorListFromTensor�
gru_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_9/strided_slice_2/stack�
gru_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_9/strided_slice_2/stack_1�
gru_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_9/strided_slice_2/stack_2�
gru_9/strided_slice_2StridedSlicegru_9/transpose:y:0$gru_9/strided_slice_2/stack:output:0&gru_9/strided_slice_2/stack_1:output:0&gru_9/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
gru_9/strided_slice_2�
gru_9/gru_cell_9/ReadVariableOpReadVariableOp(gru_9_gru_cell_9_readvariableop_resource*
_output_shapes
:	�*
dtype02!
gru_9/gru_cell_9/ReadVariableOp�
gru_9/gru_cell_9/unstackUnpack'gru_9/gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2
gru_9/gru_cell_9/unstack�
&gru_9/gru_cell_9/MatMul/ReadVariableOpReadVariableOp/gru_9_gru_cell_9_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02(
&gru_9/gru_cell_9/MatMul/ReadVariableOp�
gru_9/gru_cell_9/MatMulMatMulgru_9/strided_slice_2:output:0.gru_9/gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
gru_9/gru_cell_9/MatMul�
gru_9/gru_cell_9/BiasAddBiasAdd!gru_9/gru_cell_9/MatMul:product:0!gru_9/gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:����������2
gru_9/gru_cell_9/BiasAddr
gru_9/gru_cell_9/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_9/gru_cell_9/Const�
 gru_9/gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2"
 gru_9/gru_cell_9/split/split_dim�
gru_9/gru_cell_9/splitSplit)gru_9/gru_cell_9/split/split_dim:output:0!gru_9/gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
gru_9/gru_cell_9/split�
(gru_9/gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp1gru_9_gru_cell_9_matmul_1_readvariableop_resource*
_output_shapes
:	K�*
dtype02*
(gru_9/gru_cell_9/MatMul_1/ReadVariableOp�
gru_9/gru_cell_9/MatMul_1MatMulgru_9/zeros:output:00gru_9/gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
gru_9/gru_cell_9/MatMul_1�
gru_9/gru_cell_9/BiasAdd_1BiasAdd#gru_9/gru_cell_9/MatMul_1:product:0!gru_9/gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:����������2
gru_9/gru_cell_9/BiasAdd_1�
gru_9/gru_cell_9/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"K   K   ����2
gru_9/gru_cell_9/Const_1�
"gru_9/gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2$
"gru_9/gru_cell_9/split_1/split_dim�
gru_9/gru_cell_9/split_1SplitV#gru_9/gru_cell_9/BiasAdd_1:output:0!gru_9/gru_cell_9/Const_1:output:0+gru_9/gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
gru_9/gru_cell_9/split_1�
gru_9/gru_cell_9/addAddV2gru_9/gru_cell_9/split:output:0!gru_9/gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:���������K2
gru_9/gru_cell_9/add�
gru_9/gru_cell_9/SigmoidSigmoidgru_9/gru_cell_9/add:z:0*
T0*'
_output_shapes
:���������K2
gru_9/gru_cell_9/Sigmoid�
gru_9/gru_cell_9/add_1AddV2gru_9/gru_cell_9/split:output:1!gru_9/gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:���������K2
gru_9/gru_cell_9/add_1�
gru_9/gru_cell_9/Sigmoid_1Sigmoidgru_9/gru_cell_9/add_1:z:0*
T0*'
_output_shapes
:���������K2
gru_9/gru_cell_9/Sigmoid_1�
gru_9/gru_cell_9/mulMulgru_9/gru_cell_9/Sigmoid_1:y:0!gru_9/gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:���������K2
gru_9/gru_cell_9/mul�
gru_9/gru_cell_9/add_2AddV2gru_9/gru_cell_9/split:output:2gru_9/gru_cell_9/mul:z:0*
T0*'
_output_shapes
:���������K2
gru_9/gru_cell_9/add_2�
gru_9/gru_cell_9/ReluRelugru_9/gru_cell_9/add_2:z:0*
T0*'
_output_shapes
:���������K2
gru_9/gru_cell_9/Relu�
gru_9/gru_cell_9/mul_1Mulgru_9/gru_cell_9/Sigmoid:y:0gru_9/zeros:output:0*
T0*'
_output_shapes
:���������K2
gru_9/gru_cell_9/mul_1u
gru_9/gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
gru_9/gru_cell_9/sub/x�
gru_9/gru_cell_9/subSubgru_9/gru_cell_9/sub/x:output:0gru_9/gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:���������K2
gru_9/gru_cell_9/sub�
gru_9/gru_cell_9/mul_2Mulgru_9/gru_cell_9/sub:z:0#gru_9/gru_cell_9/Relu:activations:0*
T0*'
_output_shapes
:���������K2
gru_9/gru_cell_9/mul_2�
gru_9/gru_cell_9/add_3AddV2gru_9/gru_cell_9/mul_1:z:0gru_9/gru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:���������K2
gru_9/gru_cell_9/add_3�
#gru_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����K   2%
#gru_9/TensorArrayV2_1/element_shape�
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

gru_9/time�
gru_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2 
gru_9/while/maximum_iterationsv
gru_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_9/while/loop_counter�
gru_9/whileWhile!gru_9/while/loop_counter:output:0'gru_9/while/maximum_iterations:output:0gru_9/time:output:0gru_9/TensorArrayV2_1:handle:0gru_9/zeros:output:0gru_9/strided_slice_1:output:0=gru_9/TensorArrayUnstack/TensorListFromTensor:output_handle:0(gru_9_gru_cell_9_readvariableop_resource/gru_9_gru_cell_9_matmul_readvariableop_resource1gru_9_gru_cell_9_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������K: : : : : *%
_read_only_resource_inputs
	*%
bodyR
gru_9_while_body_27549639*%
condR
gru_9_while_cond_27549638*8
output_shapes'
%: : : : :���������K: : : : : *
parallel_iterations 2
gru_9/while�
6gru_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����K   28
6gru_9/TensorArrayV2Stack/TensorListStack/element_shape�
(gru_9/TensorArrayV2Stack/TensorListStackTensorListStackgru_9/while:output:3?gru_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������K*
element_dtype02*
(gru_9/TensorArrayV2Stack/TensorListStack�
gru_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
gru_9/strided_slice_3/stack�
gru_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
gru_9/strided_slice_3/stack_1�
gru_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_9/strided_slice_3/stack_2�
gru_9/strided_slice_3StridedSlice1gru_9/TensorArrayV2Stack/TensorListStack:tensor:0$gru_9/strided_slice_3/stack:output:0&gru_9/strided_slice_3/stack_1:output:0&gru_9/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������K*
shrink_axis_mask2
gru_9/strided_slice_3�
gru_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_9/transpose_1/perm�
gru_9/transpose_1	Transpose1gru_9/TensorArrayV2Stack/TensorListStack:tensor:0gru_9/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������K2
gru_9/transpose_1r
gru_9/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_9/runtime�
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes

:Kd*
dtype02 
dense_18/MatMul/ReadVariableOp�
dense_18/MatMulMatMulgru_9/strided_slice_3:output:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
dense_18/MatMul�
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_18/BiasAdd/ReadVariableOp�
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
dense_18/BiasAdds
dense_18/ReluReludense_18/BiasAdd:output:0*
T0*'
_output_shapes
:���������d2
dense_18/Relu�
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02 
dense_19/MatMul/ReadVariableOp�
dense_19/MatMulMatMuldense_18/Relu:activations:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_19/MatMul�
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_19/BiasAdd/ReadVariableOp�
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_19/BiasAdd�
IdentityIdentitydense_19/BiasAdd:output:0 ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp'^gru_9/gru_cell_9/MatMul/ReadVariableOp)^gru_9/gru_cell_9/MatMul_1/ReadVariableOp ^gru_9/gru_cell_9/ReadVariableOp^gru_9/while*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������:::::::2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp2P
&gru_9/gru_cell_9/MatMul/ReadVariableOp&gru_9/gru_cell_9/MatMul/ReadVariableOp2T
(gru_9/gru_cell_9/MatMul_1/ReadVariableOp(gru_9/gru_cell_9/MatMul_1/ReadVariableOp2B
gru_9/gru_cell_9/ReadVariableOpgru_9/gru_cell_9/ReadVariableOp2
gru_9/whilegru_9/while:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
/__inference_sequential_9_layer_call_fn_27549501
gru_9_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallgru_9_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_sequential_9_layer_call_and_return_conditional_losses_275494842
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������:::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:���������
%
_user_specified_namegru_9_input
�
�
while_cond_27550020
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_27550020___redundant_placeholder06
2while_while_cond_27550020___redundant_placeholder16
2while_while_cond_27550020___redundant_placeholder26
2while_while_cond_27550020___redundant_placeholder3
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
-: : : : :���������K: ::::: 
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
:���������K:

_output_shapes
: :

_output_shapes
:
�
�
(__inference_gru_9_layer_call_fn_27550292
inputs_0
unknown
	unknown_0
	unknown_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������K*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_gru_9_layer_call_and_return_conditional_losses_275490252
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������K2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�
�
H__inference_gru_cell_9_layer_call_and_return_conditional_losses_27548584

inputs

states
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	�*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2	
unstack�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:����������2	
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
���������2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
split�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	K�*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:����������2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"K   K   ����2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split_1/split_dim�
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:���������K2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������K2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:���������K2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������K2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:���������K2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:���������K2
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:���������K2
Relu\
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:���������K2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������K2
subd
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:���������K2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:���������K2
add_3�
IdentityIdentity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:���������K2

Identity�

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:���������K2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:���������:���������K:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������K
 
_user_specified_namestates
�
�
&sequential_9_gru_9_while_cond_27548368B
>sequential_9_gru_9_while_sequential_9_gru_9_while_loop_counterH
Dsequential_9_gru_9_while_sequential_9_gru_9_while_maximum_iterations(
$sequential_9_gru_9_while_placeholder*
&sequential_9_gru_9_while_placeholder_1*
&sequential_9_gru_9_while_placeholder_2D
@sequential_9_gru_9_while_less_sequential_9_gru_9_strided_slice_1\
Xsequential_9_gru_9_while_sequential_9_gru_9_while_cond_27548368___redundant_placeholder0\
Xsequential_9_gru_9_while_sequential_9_gru_9_while_cond_27548368___redundant_placeholder1\
Xsequential_9_gru_9_while_sequential_9_gru_9_while_cond_27548368___redundant_placeholder2\
Xsequential_9_gru_9_while_sequential_9_gru_9_while_cond_27548368___redundant_placeholder3%
!sequential_9_gru_9_while_identity
�
sequential_9/gru_9/while/LessLess$sequential_9_gru_9_while_placeholder@sequential_9_gru_9_while_less_sequential_9_gru_9_strided_slice_1*
T0*
_output_shapes
: 2
sequential_9/gru_9/while/Less�
!sequential_9/gru_9/while/IdentityIdentity!sequential_9/gru_9/while/Less:z:0*
T0
*
_output_shapes
: 2#
!sequential_9/gru_9/while/Identity"O
!sequential_9_gru_9_while_identity*sequential_9/gru_9/while/Identity:output:0*@
_input_shapes/
-: : : : :���������K: ::::: 
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
:���������K:

_output_shapes
: :

_output_shapes
:
�F
�
while_body_27550520
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
1while_gru_cell_9_matmul_1_readvariableop_resource��&while/gru_cell_9/MatMul/ReadVariableOp�(while/gru_cell_9/MatMul_1/ReadVariableOp�while/gru_cell_9/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
while/gru_cell_9/ReadVariableOpReadVariableOp*while_gru_cell_9_readvariableop_resource_0*
_output_shapes
:	�*
dtype02!
while/gru_cell_9/ReadVariableOp�
while/gru_cell_9/unstackUnpack'while/gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2
while/gru_cell_9/unstack�
&while/gru_cell_9/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02(
&while/gru_cell_9/MatMul/ReadVariableOp�
while/gru_cell_9/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/gru_cell_9/MatMul�
while/gru_cell_9/BiasAddBiasAdd!while/gru_cell_9/MatMul:product:0!while/gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:����������2
while/gru_cell_9/BiasAddr
while/gru_cell_9/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_9/Const�
 while/gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2"
 while/gru_cell_9/split/split_dim�
while/gru_cell_9/splitSplit)while/gru_cell_9/split/split_dim:output:0!while/gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
while/gru_cell_9/split�
(while/gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes
:	K�*
dtype02*
(while/gru_cell_9/MatMul_1/ReadVariableOp�
while/gru_cell_9/MatMul_1MatMulwhile_placeholder_20while/gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/gru_cell_9/MatMul_1�
while/gru_cell_9/BiasAdd_1BiasAdd#while/gru_cell_9/MatMul_1:product:0!while/gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:����������2
while/gru_cell_9/BiasAdd_1�
while/gru_cell_9/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"K   K   ����2
while/gru_cell_9/Const_1�
"while/gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2$
"while/gru_cell_9/split_1/split_dim�
while/gru_cell_9/split_1SplitV#while/gru_cell_9/BiasAdd_1:output:0!while/gru_cell_9/Const_1:output:0+while/gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
while/gru_cell_9/split_1�
while/gru_cell_9/addAddV2while/gru_cell_9/split:output:0!while/gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/add�
while/gru_cell_9/SigmoidSigmoidwhile/gru_cell_9/add:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/Sigmoid�
while/gru_cell_9/add_1AddV2while/gru_cell_9/split:output:1!while/gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/add_1�
while/gru_cell_9/Sigmoid_1Sigmoidwhile/gru_cell_9/add_1:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/Sigmoid_1�
while/gru_cell_9/mulMulwhile/gru_cell_9/Sigmoid_1:y:0!while/gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/mul�
while/gru_cell_9/add_2AddV2while/gru_cell_9/split:output:2while/gru_cell_9/mul:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/add_2�
while/gru_cell_9/ReluReluwhile/gru_cell_9/add_2:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/Relu�
while/gru_cell_9/mul_1Mulwhile/gru_cell_9/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/mul_1u
while/gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
while/gru_cell_9/sub/x�
while/gru_cell_9/subSubwhile/gru_cell_9/sub/x:output:0while/gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/sub�
while/gru_cell_9/mul_2Mulwhile/gru_cell_9/sub:z:0#while/gru_cell_9/Relu:activations:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/mul_2�
while/gru_cell_9/add_3AddV2while/gru_cell_9/mul_1:z:0while/gru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/add_3�
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
while/add_1�
while/IdentityIdentitywhile/add_1:z:0'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/gru_cell_9/add_3:z:0'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*'
_output_shapes
:���������K2
while/Identity_4"h
1while_gru_cell_9_matmul_1_readvariableop_resource3while_gru_cell_9_matmul_1_readvariableop_resource_0"d
/while_gru_cell_9_matmul_readvariableop_resource1while_gru_cell_9_matmul_readvariableop_resource_0"V
(while_gru_cell_9_readvariableop_resource*while_gru_cell_9_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :���������K: : :::2P
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
:���������K:

_output_shapes
: :

_output_shapes
: 
�	
�
F__inference_dense_18_layer_call_and_return_conditional_losses_27549396

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Kd*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������d2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������K::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������K
 
_user_specified_nameinputs
�w
�
J__inference_sequential_9_layer_call_and_return_conditional_losses_27549914

inputs,
(gru_9_gru_cell_9_readvariableop_resource3
/gru_9_gru_cell_9_matmul_readvariableop_resource5
1gru_9_gru_cell_9_matmul_1_readvariableop_resource+
'dense_18_matmul_readvariableop_resource,
(dense_18_biasadd_readvariableop_resource+
'dense_19_matmul_readvariableop_resource,
(dense_19_biasadd_readvariableop_resource
identity��dense_18/BiasAdd/ReadVariableOp�dense_18/MatMul/ReadVariableOp�dense_19/BiasAdd/ReadVariableOp�dense_19/MatMul/ReadVariableOp�&gru_9/gru_cell_9/MatMul/ReadVariableOp�(gru_9/gru_cell_9/MatMul_1/ReadVariableOp�gru_9/gru_cell_9/ReadVariableOp�gru_9/whileP
gru_9/ShapeShapeinputs*
T0*
_output_shapes
:2
gru_9/Shape�
gru_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_9/strided_slice/stack�
gru_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_9/strided_slice/stack_1�
gru_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_9/strided_slice/stack_2�
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
value	B :K2
gru_9/zeros/mul/y�
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
B :�2
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
value	B :K2
gru_9/zeros/packed/1�
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
gru_9/zeros/Const�
gru_9/zerosFillgru_9/zeros/packed:output:0gru_9/zeros/Const:output:0*
T0*'
_output_shapes
:���������K2
gru_9/zeros�
gru_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_9/transpose/perm�
gru_9/transpose	Transposeinputsgru_9/transpose/perm:output:0*
T0*+
_output_shapes
:���������2
gru_9/transposea
gru_9/Shape_1Shapegru_9/transpose:y:0*
T0*
_output_shapes
:2
gru_9/Shape_1�
gru_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_9/strided_slice_1/stack�
gru_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_9/strided_slice_1/stack_1�
gru_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_9/strided_slice_1/stack_2�
gru_9/strided_slice_1StridedSlicegru_9/Shape_1:output:0$gru_9/strided_slice_1/stack:output:0&gru_9/strided_slice_1/stack_1:output:0&gru_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_9/strided_slice_1�
!gru_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2#
!gru_9/TensorArrayV2/element_shape�
gru_9/TensorArrayV2TensorListReserve*gru_9/TensorArrayV2/element_shape:output:0gru_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_9/TensorArrayV2�
;gru_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2=
;gru_9/TensorArrayUnstack/TensorListFromTensor/element_shape�
-gru_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_9/transpose:y:0Dgru_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02/
-gru_9/TensorArrayUnstack/TensorListFromTensor�
gru_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_9/strided_slice_2/stack�
gru_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_9/strided_slice_2/stack_1�
gru_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_9/strided_slice_2/stack_2�
gru_9/strided_slice_2StridedSlicegru_9/transpose:y:0$gru_9/strided_slice_2/stack:output:0&gru_9/strided_slice_2/stack_1:output:0&gru_9/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
gru_9/strided_slice_2�
gru_9/gru_cell_9/ReadVariableOpReadVariableOp(gru_9_gru_cell_9_readvariableop_resource*
_output_shapes
:	�*
dtype02!
gru_9/gru_cell_9/ReadVariableOp�
gru_9/gru_cell_9/unstackUnpack'gru_9/gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2
gru_9/gru_cell_9/unstack�
&gru_9/gru_cell_9/MatMul/ReadVariableOpReadVariableOp/gru_9_gru_cell_9_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02(
&gru_9/gru_cell_9/MatMul/ReadVariableOp�
gru_9/gru_cell_9/MatMulMatMulgru_9/strided_slice_2:output:0.gru_9/gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
gru_9/gru_cell_9/MatMul�
gru_9/gru_cell_9/BiasAddBiasAdd!gru_9/gru_cell_9/MatMul:product:0!gru_9/gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:����������2
gru_9/gru_cell_9/BiasAddr
gru_9/gru_cell_9/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_9/gru_cell_9/Const�
 gru_9/gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2"
 gru_9/gru_cell_9/split/split_dim�
gru_9/gru_cell_9/splitSplit)gru_9/gru_cell_9/split/split_dim:output:0!gru_9/gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
gru_9/gru_cell_9/split�
(gru_9/gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp1gru_9_gru_cell_9_matmul_1_readvariableop_resource*
_output_shapes
:	K�*
dtype02*
(gru_9/gru_cell_9/MatMul_1/ReadVariableOp�
gru_9/gru_cell_9/MatMul_1MatMulgru_9/zeros:output:00gru_9/gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
gru_9/gru_cell_9/MatMul_1�
gru_9/gru_cell_9/BiasAdd_1BiasAdd#gru_9/gru_cell_9/MatMul_1:product:0!gru_9/gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:����������2
gru_9/gru_cell_9/BiasAdd_1�
gru_9/gru_cell_9/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"K   K   ����2
gru_9/gru_cell_9/Const_1�
"gru_9/gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2$
"gru_9/gru_cell_9/split_1/split_dim�
gru_9/gru_cell_9/split_1SplitV#gru_9/gru_cell_9/BiasAdd_1:output:0!gru_9/gru_cell_9/Const_1:output:0+gru_9/gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
gru_9/gru_cell_9/split_1�
gru_9/gru_cell_9/addAddV2gru_9/gru_cell_9/split:output:0!gru_9/gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:���������K2
gru_9/gru_cell_9/add�
gru_9/gru_cell_9/SigmoidSigmoidgru_9/gru_cell_9/add:z:0*
T0*'
_output_shapes
:���������K2
gru_9/gru_cell_9/Sigmoid�
gru_9/gru_cell_9/add_1AddV2gru_9/gru_cell_9/split:output:1!gru_9/gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:���������K2
gru_9/gru_cell_9/add_1�
gru_9/gru_cell_9/Sigmoid_1Sigmoidgru_9/gru_cell_9/add_1:z:0*
T0*'
_output_shapes
:���������K2
gru_9/gru_cell_9/Sigmoid_1�
gru_9/gru_cell_9/mulMulgru_9/gru_cell_9/Sigmoid_1:y:0!gru_9/gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:���������K2
gru_9/gru_cell_9/mul�
gru_9/gru_cell_9/add_2AddV2gru_9/gru_cell_9/split:output:2gru_9/gru_cell_9/mul:z:0*
T0*'
_output_shapes
:���������K2
gru_9/gru_cell_9/add_2�
gru_9/gru_cell_9/ReluRelugru_9/gru_cell_9/add_2:z:0*
T0*'
_output_shapes
:���������K2
gru_9/gru_cell_9/Relu�
gru_9/gru_cell_9/mul_1Mulgru_9/gru_cell_9/Sigmoid:y:0gru_9/zeros:output:0*
T0*'
_output_shapes
:���������K2
gru_9/gru_cell_9/mul_1u
gru_9/gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
gru_9/gru_cell_9/sub/x�
gru_9/gru_cell_9/subSubgru_9/gru_cell_9/sub/x:output:0gru_9/gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:���������K2
gru_9/gru_cell_9/sub�
gru_9/gru_cell_9/mul_2Mulgru_9/gru_cell_9/sub:z:0#gru_9/gru_cell_9/Relu:activations:0*
T0*'
_output_shapes
:���������K2
gru_9/gru_cell_9/mul_2�
gru_9/gru_cell_9/add_3AddV2gru_9/gru_cell_9/mul_1:z:0gru_9/gru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:���������K2
gru_9/gru_cell_9/add_3�
#gru_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����K   2%
#gru_9/TensorArrayV2_1/element_shape�
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

gru_9/time�
gru_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2 
gru_9/while/maximum_iterationsv
gru_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_9/while/loop_counter�
gru_9/whileWhile!gru_9/while/loop_counter:output:0'gru_9/while/maximum_iterations:output:0gru_9/time:output:0gru_9/TensorArrayV2_1:handle:0gru_9/zeros:output:0gru_9/strided_slice_1:output:0=gru_9/TensorArrayUnstack/TensorListFromTensor:output_handle:0(gru_9_gru_cell_9_readvariableop_resource/gru_9_gru_cell_9_matmul_readvariableop_resource1gru_9_gru_cell_9_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������K: : : : : *%
_read_only_resource_inputs
	*%
bodyR
gru_9_while_body_27549811*%
condR
gru_9_while_cond_27549810*8
output_shapes'
%: : : : :���������K: : : : : *
parallel_iterations 2
gru_9/while�
6gru_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����K   28
6gru_9/TensorArrayV2Stack/TensorListStack/element_shape�
(gru_9/TensorArrayV2Stack/TensorListStackTensorListStackgru_9/while:output:3?gru_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������K*
element_dtype02*
(gru_9/TensorArrayV2Stack/TensorListStack�
gru_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
gru_9/strided_slice_3/stack�
gru_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
gru_9/strided_slice_3/stack_1�
gru_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_9/strided_slice_3/stack_2�
gru_9/strided_slice_3StridedSlice1gru_9/TensorArrayV2Stack/TensorListStack:tensor:0$gru_9/strided_slice_3/stack:output:0&gru_9/strided_slice_3/stack_1:output:0&gru_9/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������K*
shrink_axis_mask2
gru_9/strided_slice_3�
gru_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_9/transpose_1/perm�
gru_9/transpose_1	Transpose1gru_9/TensorArrayV2Stack/TensorListStack:tensor:0gru_9/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������K2
gru_9/transpose_1r
gru_9/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_9/runtime�
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes

:Kd*
dtype02 
dense_18/MatMul/ReadVariableOp�
dense_18/MatMulMatMulgru_9/strided_slice_3:output:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
dense_18/MatMul�
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_18/BiasAdd/ReadVariableOp�
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
dense_18/BiasAdds
dense_18/ReluReludense_18/BiasAdd:output:0*
T0*'
_output_shapes
:���������d2
dense_18/Relu�
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02 
dense_19/MatMul/ReadVariableOp�
dense_19/MatMulMatMuldense_18/Relu:activations:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_19/MatMul�
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_19/BiasAdd/ReadVariableOp�
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_19/BiasAdd�
IdentityIdentitydense_19/BiasAdd:output:0 ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp'^gru_9/gru_cell_9/MatMul/ReadVariableOp)^gru_9/gru_cell_9/MatMul_1/ReadVariableOp ^gru_9/gru_cell_9/ReadVariableOp^gru_9/while*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������:::::::2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp2P
&gru_9/gru_cell_9/MatMul/ReadVariableOp&gru_9/gru_cell_9/MatMul/ReadVariableOp2T
(gru_9/gru_cell_9/MatMul_1/ReadVariableOp(gru_9/gru_cell_9/MatMul_1/ReadVariableOp2B
gru_9/gru_cell_9/ReadVariableOpgru_9/gru_cell_9/ReadVariableOp2
gru_9/whilegru_9/while:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�!
�
while_body_27548961
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_gru_cell_9_27548983_0
while_gru_cell_9_27548985_0
while_gru_cell_9_27548987_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_gru_cell_9_27548983
while_gru_cell_9_27548985
while_gru_cell_9_27548987��(while/gru_cell_9/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
(while/gru_cell_9/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_9_27548983_0while_gru_cell_9_27548985_0while_gru_cell_9_27548987_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������K:���������K*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_gru_cell_9_layer_call_and_return_conditional_losses_275485842*
(while/gru_cell_9/StatefulPartitionedCall�
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
while/add_1�
while/IdentityIdentitywhile/add_1:z:0)^while/gru_cell_9/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations)^while/gru_cell_9/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0)^while/gru_cell_9/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/gru_cell_9/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity1while/gru_cell_9/StatefulPartitionedCall:output:1)^while/gru_cell_9/StatefulPartitionedCall*
T0*'
_output_shapes
:���������K2
while/Identity_4"8
while_gru_cell_9_27548983while_gru_cell_9_27548983_0"8
while_gru_cell_9_27548985while_gru_cell_9_27548985_0"8
while_gru_cell_9_27548987while_gru_cell_9_27548987_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :���������K: : :::2T
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
:���������K:

_output_shapes
: :

_output_shapes
: 
�
�
while_cond_27548842
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_27548842___redundant_placeholder06
2while_while_cond_27548842___redundant_placeholder16
2while_while_cond_27548842___redundant_placeholder26
2while_while_cond_27548842___redundant_placeholder3
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
-: : : : :���������K: ::::: 
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
:���������K:

_output_shapes
: :

_output_shapes
:
�
�
#__inference__wrapped_model_27548472
gru_9_input9
5sequential_9_gru_9_gru_cell_9_readvariableop_resource@
<sequential_9_gru_9_gru_cell_9_matmul_readvariableop_resourceB
>sequential_9_gru_9_gru_cell_9_matmul_1_readvariableop_resource8
4sequential_9_dense_18_matmul_readvariableop_resource9
5sequential_9_dense_18_biasadd_readvariableop_resource8
4sequential_9_dense_19_matmul_readvariableop_resource9
5sequential_9_dense_19_biasadd_readvariableop_resource
identity��,sequential_9/dense_18/BiasAdd/ReadVariableOp�+sequential_9/dense_18/MatMul/ReadVariableOp�,sequential_9/dense_19/BiasAdd/ReadVariableOp�+sequential_9/dense_19/MatMul/ReadVariableOp�3sequential_9/gru_9/gru_cell_9/MatMul/ReadVariableOp�5sequential_9/gru_9/gru_cell_9/MatMul_1/ReadVariableOp�,sequential_9/gru_9/gru_cell_9/ReadVariableOp�sequential_9/gru_9/whileo
sequential_9/gru_9/ShapeShapegru_9_input*
T0*
_output_shapes
:2
sequential_9/gru_9/Shape�
&sequential_9/gru_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_9/gru_9/strided_slice/stack�
(sequential_9/gru_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential_9/gru_9/strided_slice/stack_1�
(sequential_9/gru_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential_9/gru_9/strided_slice/stack_2�
 sequential_9/gru_9/strided_sliceStridedSlice!sequential_9/gru_9/Shape:output:0/sequential_9/gru_9/strided_slice/stack:output:01sequential_9/gru_9/strided_slice/stack_1:output:01sequential_9/gru_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 sequential_9/gru_9/strided_slice�
sequential_9/gru_9/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :K2 
sequential_9/gru_9/zeros/mul/y�
sequential_9/gru_9/zeros/mulMul)sequential_9/gru_9/strided_slice:output:0'sequential_9/gru_9/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_9/gru_9/zeros/mul�
sequential_9/gru_9/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2!
sequential_9/gru_9/zeros/Less/y�
sequential_9/gru_9/zeros/LessLess sequential_9/gru_9/zeros/mul:z:0(sequential_9/gru_9/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
sequential_9/gru_9/zeros/Less�
!sequential_9/gru_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K2#
!sequential_9/gru_9/zeros/packed/1�
sequential_9/gru_9/zeros/packedPack)sequential_9/gru_9/strided_slice:output:0*sequential_9/gru_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2!
sequential_9/gru_9/zeros/packed�
sequential_9/gru_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
sequential_9/gru_9/zeros/Const�
sequential_9/gru_9/zerosFill(sequential_9/gru_9/zeros/packed:output:0'sequential_9/gru_9/zeros/Const:output:0*
T0*'
_output_shapes
:���������K2
sequential_9/gru_9/zeros�
!sequential_9/gru_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!sequential_9/gru_9/transpose/perm�
sequential_9/gru_9/transpose	Transposegru_9_input*sequential_9/gru_9/transpose/perm:output:0*
T0*+
_output_shapes
:���������2
sequential_9/gru_9/transpose�
sequential_9/gru_9/Shape_1Shape sequential_9/gru_9/transpose:y:0*
T0*
_output_shapes
:2
sequential_9/gru_9/Shape_1�
(sequential_9/gru_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_9/gru_9/strided_slice_1/stack�
*sequential_9/gru_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_9/gru_9/strided_slice_1/stack_1�
*sequential_9/gru_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_9/gru_9/strided_slice_1/stack_2�
"sequential_9/gru_9/strided_slice_1StridedSlice#sequential_9/gru_9/Shape_1:output:01sequential_9/gru_9/strided_slice_1/stack:output:03sequential_9/gru_9/strided_slice_1/stack_1:output:03sequential_9/gru_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_9/gru_9/strided_slice_1�
.sequential_9/gru_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������20
.sequential_9/gru_9/TensorArrayV2/element_shape�
 sequential_9/gru_9/TensorArrayV2TensorListReserve7sequential_9/gru_9/TensorArrayV2/element_shape:output:0+sequential_9/gru_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 sequential_9/gru_9/TensorArrayV2�
Hsequential_9/gru_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2J
Hsequential_9/gru_9/TensorArrayUnstack/TensorListFromTensor/element_shape�
:sequential_9/gru_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor sequential_9/gru_9/transpose:y:0Qsequential_9/gru_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02<
:sequential_9/gru_9/TensorArrayUnstack/TensorListFromTensor�
(sequential_9/gru_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_9/gru_9/strided_slice_2/stack�
*sequential_9/gru_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_9/gru_9/strided_slice_2/stack_1�
*sequential_9/gru_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_9/gru_9/strided_slice_2/stack_2�
"sequential_9/gru_9/strided_slice_2StridedSlice sequential_9/gru_9/transpose:y:01sequential_9/gru_9/strided_slice_2/stack:output:03sequential_9/gru_9/strided_slice_2/stack_1:output:03sequential_9/gru_9/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2$
"sequential_9/gru_9/strided_slice_2�
,sequential_9/gru_9/gru_cell_9/ReadVariableOpReadVariableOp5sequential_9_gru_9_gru_cell_9_readvariableop_resource*
_output_shapes
:	�*
dtype02.
,sequential_9/gru_9/gru_cell_9/ReadVariableOp�
%sequential_9/gru_9/gru_cell_9/unstackUnpack4sequential_9/gru_9/gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2'
%sequential_9/gru_9/gru_cell_9/unstack�
3sequential_9/gru_9/gru_cell_9/MatMul/ReadVariableOpReadVariableOp<sequential_9_gru_9_gru_cell_9_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype025
3sequential_9/gru_9/gru_cell_9/MatMul/ReadVariableOp�
$sequential_9/gru_9/gru_cell_9/MatMulMatMul+sequential_9/gru_9/strided_slice_2:output:0;sequential_9/gru_9/gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2&
$sequential_9/gru_9/gru_cell_9/MatMul�
%sequential_9/gru_9/gru_cell_9/BiasAddBiasAdd.sequential_9/gru_9/gru_cell_9/MatMul:product:0.sequential_9/gru_9/gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:����������2'
%sequential_9/gru_9/gru_cell_9/BiasAdd�
#sequential_9/gru_9/gru_cell_9/ConstConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_9/gru_9/gru_cell_9/Const�
-sequential_9/gru_9/gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2/
-sequential_9/gru_9/gru_cell_9/split/split_dim�
#sequential_9/gru_9/gru_cell_9/splitSplit6sequential_9/gru_9/gru_cell_9/split/split_dim:output:0.sequential_9/gru_9/gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2%
#sequential_9/gru_9/gru_cell_9/split�
5sequential_9/gru_9/gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp>sequential_9_gru_9_gru_cell_9_matmul_1_readvariableop_resource*
_output_shapes
:	K�*
dtype027
5sequential_9/gru_9/gru_cell_9/MatMul_1/ReadVariableOp�
&sequential_9/gru_9/gru_cell_9/MatMul_1MatMul!sequential_9/gru_9/zeros:output:0=sequential_9/gru_9/gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2(
&sequential_9/gru_9/gru_cell_9/MatMul_1�
'sequential_9/gru_9/gru_cell_9/BiasAdd_1BiasAdd0sequential_9/gru_9/gru_cell_9/MatMul_1:product:0.sequential_9/gru_9/gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:����������2)
'sequential_9/gru_9/gru_cell_9/BiasAdd_1�
%sequential_9/gru_9/gru_cell_9/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"K   K   ����2'
%sequential_9/gru_9/gru_cell_9/Const_1�
/sequential_9/gru_9/gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������21
/sequential_9/gru_9/gru_cell_9/split_1/split_dim�
%sequential_9/gru_9/gru_cell_9/split_1SplitV0sequential_9/gru_9/gru_cell_9/BiasAdd_1:output:0.sequential_9/gru_9/gru_cell_9/Const_1:output:08sequential_9/gru_9/gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2'
%sequential_9/gru_9/gru_cell_9/split_1�
!sequential_9/gru_9/gru_cell_9/addAddV2,sequential_9/gru_9/gru_cell_9/split:output:0.sequential_9/gru_9/gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:���������K2#
!sequential_9/gru_9/gru_cell_9/add�
%sequential_9/gru_9/gru_cell_9/SigmoidSigmoid%sequential_9/gru_9/gru_cell_9/add:z:0*
T0*'
_output_shapes
:���������K2'
%sequential_9/gru_9/gru_cell_9/Sigmoid�
#sequential_9/gru_9/gru_cell_9/add_1AddV2,sequential_9/gru_9/gru_cell_9/split:output:1.sequential_9/gru_9/gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:���������K2%
#sequential_9/gru_9/gru_cell_9/add_1�
'sequential_9/gru_9/gru_cell_9/Sigmoid_1Sigmoid'sequential_9/gru_9/gru_cell_9/add_1:z:0*
T0*'
_output_shapes
:���������K2)
'sequential_9/gru_9/gru_cell_9/Sigmoid_1�
!sequential_9/gru_9/gru_cell_9/mulMul+sequential_9/gru_9/gru_cell_9/Sigmoid_1:y:0.sequential_9/gru_9/gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:���������K2#
!sequential_9/gru_9/gru_cell_9/mul�
#sequential_9/gru_9/gru_cell_9/add_2AddV2,sequential_9/gru_9/gru_cell_9/split:output:2%sequential_9/gru_9/gru_cell_9/mul:z:0*
T0*'
_output_shapes
:���������K2%
#sequential_9/gru_9/gru_cell_9/add_2�
"sequential_9/gru_9/gru_cell_9/ReluRelu'sequential_9/gru_9/gru_cell_9/add_2:z:0*
T0*'
_output_shapes
:���������K2$
"sequential_9/gru_9/gru_cell_9/Relu�
#sequential_9/gru_9/gru_cell_9/mul_1Mul)sequential_9/gru_9/gru_cell_9/Sigmoid:y:0!sequential_9/gru_9/zeros:output:0*
T0*'
_output_shapes
:���������K2%
#sequential_9/gru_9/gru_cell_9/mul_1�
#sequential_9/gru_9/gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2%
#sequential_9/gru_9/gru_cell_9/sub/x�
!sequential_9/gru_9/gru_cell_9/subSub,sequential_9/gru_9/gru_cell_9/sub/x:output:0)sequential_9/gru_9/gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:���������K2#
!sequential_9/gru_9/gru_cell_9/sub�
#sequential_9/gru_9/gru_cell_9/mul_2Mul%sequential_9/gru_9/gru_cell_9/sub:z:00sequential_9/gru_9/gru_cell_9/Relu:activations:0*
T0*'
_output_shapes
:���������K2%
#sequential_9/gru_9/gru_cell_9/mul_2�
#sequential_9/gru_9/gru_cell_9/add_3AddV2'sequential_9/gru_9/gru_cell_9/mul_1:z:0'sequential_9/gru_9/gru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:���������K2%
#sequential_9/gru_9/gru_cell_9/add_3�
0sequential_9/gru_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����K   22
0sequential_9/gru_9/TensorArrayV2_1/element_shape�
"sequential_9/gru_9/TensorArrayV2_1TensorListReserve9sequential_9/gru_9/TensorArrayV2_1/element_shape:output:0+sequential_9/gru_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"sequential_9/gru_9/TensorArrayV2_1t
sequential_9/gru_9/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_9/gru_9/time�
+sequential_9/gru_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2-
+sequential_9/gru_9/while/maximum_iterations�
%sequential_9/gru_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2'
%sequential_9/gru_9/while/loop_counter�
sequential_9/gru_9/whileWhile.sequential_9/gru_9/while/loop_counter:output:04sequential_9/gru_9/while/maximum_iterations:output:0 sequential_9/gru_9/time:output:0+sequential_9/gru_9/TensorArrayV2_1:handle:0!sequential_9/gru_9/zeros:output:0+sequential_9/gru_9/strided_slice_1:output:0Jsequential_9/gru_9/TensorArrayUnstack/TensorListFromTensor:output_handle:05sequential_9_gru_9_gru_cell_9_readvariableop_resource<sequential_9_gru_9_gru_cell_9_matmul_readvariableop_resource>sequential_9_gru_9_gru_cell_9_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������K: : : : : *%
_read_only_resource_inputs
	*2
body*R(
&sequential_9_gru_9_while_body_27548369*2
cond*R(
&sequential_9_gru_9_while_cond_27548368*8
output_shapes'
%: : : : :���������K: : : : : *
parallel_iterations 2
sequential_9/gru_9/while�
Csequential_9/gru_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����K   2E
Csequential_9/gru_9/TensorArrayV2Stack/TensorListStack/element_shape�
5sequential_9/gru_9/TensorArrayV2Stack/TensorListStackTensorListStack!sequential_9/gru_9/while:output:3Lsequential_9/gru_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������K*
element_dtype027
5sequential_9/gru_9/TensorArrayV2Stack/TensorListStack�
(sequential_9/gru_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2*
(sequential_9/gru_9/strided_slice_3/stack�
*sequential_9/gru_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_9/gru_9/strided_slice_3/stack_1�
*sequential_9/gru_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_9/gru_9/strided_slice_3/stack_2�
"sequential_9/gru_9/strided_slice_3StridedSlice>sequential_9/gru_9/TensorArrayV2Stack/TensorListStack:tensor:01sequential_9/gru_9/strided_slice_3/stack:output:03sequential_9/gru_9/strided_slice_3/stack_1:output:03sequential_9/gru_9/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������K*
shrink_axis_mask2$
"sequential_9/gru_9/strided_slice_3�
#sequential_9/gru_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#sequential_9/gru_9/transpose_1/perm�
sequential_9/gru_9/transpose_1	Transpose>sequential_9/gru_9/TensorArrayV2Stack/TensorListStack:tensor:0,sequential_9/gru_9/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������K2 
sequential_9/gru_9/transpose_1�
sequential_9/gru_9/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_9/gru_9/runtime�
+sequential_9/dense_18/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_18_matmul_readvariableop_resource*
_output_shapes

:Kd*
dtype02-
+sequential_9/dense_18/MatMul/ReadVariableOp�
sequential_9/dense_18/MatMulMatMul+sequential_9/gru_9/strided_slice_3:output:03sequential_9/dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
sequential_9/dense_18/MatMul�
,sequential_9/dense_18/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_18_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02.
,sequential_9/dense_18/BiasAdd/ReadVariableOp�
sequential_9/dense_18/BiasAddBiasAdd&sequential_9/dense_18/MatMul:product:04sequential_9/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
sequential_9/dense_18/BiasAdd�
sequential_9/dense_18/ReluRelu&sequential_9/dense_18/BiasAdd:output:0*
T0*'
_output_shapes
:���������d2
sequential_9/dense_18/Relu�
+sequential_9/dense_19/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_19_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02-
+sequential_9/dense_19/MatMul/ReadVariableOp�
sequential_9/dense_19/MatMulMatMul(sequential_9/dense_18/Relu:activations:03sequential_9/dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sequential_9/dense_19/MatMul�
,sequential_9/dense_19/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_9/dense_19/BiasAdd/ReadVariableOp�
sequential_9/dense_19/BiasAddBiasAdd&sequential_9/dense_19/MatMul:product:04sequential_9/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sequential_9/dense_19/BiasAdd�
IdentityIdentity&sequential_9/dense_19/BiasAdd:output:0-^sequential_9/dense_18/BiasAdd/ReadVariableOp,^sequential_9/dense_18/MatMul/ReadVariableOp-^sequential_9/dense_19/BiasAdd/ReadVariableOp,^sequential_9/dense_19/MatMul/ReadVariableOp4^sequential_9/gru_9/gru_cell_9/MatMul/ReadVariableOp6^sequential_9/gru_9/gru_cell_9/MatMul_1/ReadVariableOp-^sequential_9/gru_9/gru_cell_9/ReadVariableOp^sequential_9/gru_9/while*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������:::::::2\
,sequential_9/dense_18/BiasAdd/ReadVariableOp,sequential_9/dense_18/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_18/MatMul/ReadVariableOp+sequential_9/dense_18/MatMul/ReadVariableOp2\
,sequential_9/dense_19/BiasAdd/ReadVariableOp,sequential_9/dense_19/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_19/MatMul/ReadVariableOp+sequential_9/dense_19/MatMul/ReadVariableOp2j
3sequential_9/gru_9/gru_cell_9/MatMul/ReadVariableOp3sequential_9/gru_9/gru_cell_9/MatMul/ReadVariableOp2n
5sequential_9/gru_9/gru_cell_9/MatMul_1/ReadVariableOp5sequential_9/gru_9/gru_cell_9/MatMul_1/ReadVariableOp2\
,sequential_9/gru_9/gru_cell_9/ReadVariableOp,sequential_9/gru_9/gru_cell_9/ReadVariableOp24
sequential_9/gru_9/whilesequential_9/gru_9/while:X T
+
_output_shapes
:���������
%
_user_specified_namegru_9_input
�F
�
while_body_27550180
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
1while_gru_cell_9_matmul_1_readvariableop_resource��&while/gru_cell_9/MatMul/ReadVariableOp�(while/gru_cell_9/MatMul_1/ReadVariableOp�while/gru_cell_9/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
while/gru_cell_9/ReadVariableOpReadVariableOp*while_gru_cell_9_readvariableop_resource_0*
_output_shapes
:	�*
dtype02!
while/gru_cell_9/ReadVariableOp�
while/gru_cell_9/unstackUnpack'while/gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2
while/gru_cell_9/unstack�
&while/gru_cell_9/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02(
&while/gru_cell_9/MatMul/ReadVariableOp�
while/gru_cell_9/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/gru_cell_9/MatMul�
while/gru_cell_9/BiasAddBiasAdd!while/gru_cell_9/MatMul:product:0!while/gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:����������2
while/gru_cell_9/BiasAddr
while/gru_cell_9/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_9/Const�
 while/gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2"
 while/gru_cell_9/split/split_dim�
while/gru_cell_9/splitSplit)while/gru_cell_9/split/split_dim:output:0!while/gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
while/gru_cell_9/split�
(while/gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes
:	K�*
dtype02*
(while/gru_cell_9/MatMul_1/ReadVariableOp�
while/gru_cell_9/MatMul_1MatMulwhile_placeholder_20while/gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/gru_cell_9/MatMul_1�
while/gru_cell_9/BiasAdd_1BiasAdd#while/gru_cell_9/MatMul_1:product:0!while/gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:����������2
while/gru_cell_9/BiasAdd_1�
while/gru_cell_9/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"K   K   ����2
while/gru_cell_9/Const_1�
"while/gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2$
"while/gru_cell_9/split_1/split_dim�
while/gru_cell_9/split_1SplitV#while/gru_cell_9/BiasAdd_1:output:0!while/gru_cell_9/Const_1:output:0+while/gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
while/gru_cell_9/split_1�
while/gru_cell_9/addAddV2while/gru_cell_9/split:output:0!while/gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/add�
while/gru_cell_9/SigmoidSigmoidwhile/gru_cell_9/add:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/Sigmoid�
while/gru_cell_9/add_1AddV2while/gru_cell_9/split:output:1!while/gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/add_1�
while/gru_cell_9/Sigmoid_1Sigmoidwhile/gru_cell_9/add_1:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/Sigmoid_1�
while/gru_cell_9/mulMulwhile/gru_cell_9/Sigmoid_1:y:0!while/gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/mul�
while/gru_cell_9/add_2AddV2while/gru_cell_9/split:output:2while/gru_cell_9/mul:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/add_2�
while/gru_cell_9/ReluReluwhile/gru_cell_9/add_2:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/Relu�
while/gru_cell_9/mul_1Mulwhile/gru_cell_9/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/mul_1u
while/gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
while/gru_cell_9/sub/x�
while/gru_cell_9/subSubwhile/gru_cell_9/sub/x:output:0while/gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/sub�
while/gru_cell_9/mul_2Mulwhile/gru_cell_9/sub:z:0#while/gru_cell_9/Relu:activations:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/mul_2�
while/gru_cell_9/add_3AddV2while/gru_cell_9/mul_1:z:0while/gru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/add_3�
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
while/add_1�
while/IdentityIdentitywhile/add_1:z:0'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/gru_cell_9/add_3:z:0'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*'
_output_shapes
:���������K2
while/Identity_4"h
1while_gru_cell_9_matmul_1_readvariableop_resource3while_gru_cell_9_matmul_1_readvariableop_resource_0"d
/while_gru_cell_9_matmul_readvariableop_resource1while_gru_cell_9_matmul_readvariableop_resource_0"V
(while_gru_cell_9_readvariableop_resource*while_gru_cell_9_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :���������K: : :::2P
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
:���������K:

_output_shapes
: :

_output_shapes
: 
�
�
H__inference_gru_cell_9_layer_call_and_return_conditional_losses_27550751

inputs
states_0
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	�*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2	
unstack�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:����������2	
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
���������2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
split�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	K�*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:����������2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"K   K   ����2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split_1/split_dim�
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:���������K2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������K2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:���������K2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������K2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:���������K2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:���������K2
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:���������K2
Relu^
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:���������K2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������K2
subd
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:���������K2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:���������K2
add_3�
IdentityIdentity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:���������K2

Identity�

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:���������K2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:���������:���������K:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������K
"
_user_specified_name
states/0
�
�
while_cond_27548960
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_27548960___redundant_placeholder06
2while_while_cond_27548960___redundant_placeholder16
2while_while_cond_27548960___redundant_placeholder26
2while_while_cond_27548960___redundant_placeholder3
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
-: : : : :���������K: ::::: 
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
:���������K:

_output_shapes
: :

_output_shapes
:
�F
�
while_body_27549265
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
1while_gru_cell_9_matmul_1_readvariableop_resource��&while/gru_cell_9/MatMul/ReadVariableOp�(while/gru_cell_9/MatMul_1/ReadVariableOp�while/gru_cell_9/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
while/gru_cell_9/ReadVariableOpReadVariableOp*while_gru_cell_9_readvariableop_resource_0*
_output_shapes
:	�*
dtype02!
while/gru_cell_9/ReadVariableOp�
while/gru_cell_9/unstackUnpack'while/gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2
while/gru_cell_9/unstack�
&while/gru_cell_9/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02(
&while/gru_cell_9/MatMul/ReadVariableOp�
while/gru_cell_9/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/gru_cell_9/MatMul�
while/gru_cell_9/BiasAddBiasAdd!while/gru_cell_9/MatMul:product:0!while/gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:����������2
while/gru_cell_9/BiasAddr
while/gru_cell_9/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_9/Const�
 while/gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2"
 while/gru_cell_9/split/split_dim�
while/gru_cell_9/splitSplit)while/gru_cell_9/split/split_dim:output:0!while/gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
while/gru_cell_9/split�
(while/gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes
:	K�*
dtype02*
(while/gru_cell_9/MatMul_1/ReadVariableOp�
while/gru_cell_9/MatMul_1MatMulwhile_placeholder_20while/gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/gru_cell_9/MatMul_1�
while/gru_cell_9/BiasAdd_1BiasAdd#while/gru_cell_9/MatMul_1:product:0!while/gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:����������2
while/gru_cell_9/BiasAdd_1�
while/gru_cell_9/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"K   K   ����2
while/gru_cell_9/Const_1�
"while/gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2$
"while/gru_cell_9/split_1/split_dim�
while/gru_cell_9/split_1SplitV#while/gru_cell_9/BiasAdd_1:output:0!while/gru_cell_9/Const_1:output:0+while/gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
while/gru_cell_9/split_1�
while/gru_cell_9/addAddV2while/gru_cell_9/split:output:0!while/gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/add�
while/gru_cell_9/SigmoidSigmoidwhile/gru_cell_9/add:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/Sigmoid�
while/gru_cell_9/add_1AddV2while/gru_cell_9/split:output:1!while/gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/add_1�
while/gru_cell_9/Sigmoid_1Sigmoidwhile/gru_cell_9/add_1:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/Sigmoid_1�
while/gru_cell_9/mulMulwhile/gru_cell_9/Sigmoid_1:y:0!while/gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/mul�
while/gru_cell_9/add_2AddV2while/gru_cell_9/split:output:2while/gru_cell_9/mul:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/add_2�
while/gru_cell_9/ReluReluwhile/gru_cell_9/add_2:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/Relu�
while/gru_cell_9/mul_1Mulwhile/gru_cell_9/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/mul_1u
while/gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
while/gru_cell_9/sub/x�
while/gru_cell_9/subSubwhile/gru_cell_9/sub/x:output:0while/gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/sub�
while/gru_cell_9/mul_2Mulwhile/gru_cell_9/sub:z:0#while/gru_cell_9/Relu:activations:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/mul_2�
while/gru_cell_9/add_3AddV2while/gru_cell_9/mul_1:z:0while/gru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/add_3�
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
while/add_1�
while/IdentityIdentitywhile/add_1:z:0'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/gru_cell_9/add_3:z:0'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*'
_output_shapes
:���������K2
while/Identity_4"h
1while_gru_cell_9_matmul_1_readvariableop_resource3while_gru_cell_9_matmul_1_readvariableop_resource_0"d
/while_gru_cell_9_matmul_readvariableop_resource1while_gru_cell_9_matmul_readvariableop_resource_0"V
(while_gru_cell_9_readvariableop_resource*while_gru_cell_9_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :���������K: : :::2P
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
:���������K:

_output_shapes
: :

_output_shapes
: 
�	
�
gru_9_while_cond_27549638(
$gru_9_while_gru_9_while_loop_counter.
*gru_9_while_gru_9_while_maximum_iterations
gru_9_while_placeholder
gru_9_while_placeholder_1
gru_9_while_placeholder_2*
&gru_9_while_less_gru_9_strided_slice_1B
>gru_9_while_gru_9_while_cond_27549638___redundant_placeholder0B
>gru_9_while_gru_9_while_cond_27549638___redundant_placeholder1B
>gru_9_while_gru_9_while_cond_27549638___redundant_placeholder2B
>gru_9_while_gru_9_while_cond_27549638___redundant_placeholder3
gru_9_while_identity
�
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
-: : : : :���������K: ::::: 
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
:���������K:

_output_shapes
: :

_output_shapes
:
�Z
�
C__inference_gru_9_layer_call_and_return_conditional_losses_27550270
inputs_0&
"gru_cell_9_readvariableop_resource-
)gru_cell_9_matmul_readvariableop_resource/
+gru_cell_9_matmul_1_readvariableop_resource
identity�� gru_cell_9/MatMul/ReadVariableOp�"gru_cell_9/MatMul_1/ReadVariableOp�gru_cell_9/ReadVariableOp�whileF
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
strided_slice/stack_2�
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
B :�2
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
zeros/packed/1�
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
:���������K2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
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
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
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
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
gru_cell_9/ReadVariableOpReadVariableOp"gru_cell_9_readvariableop_resource*
_output_shapes
:	�*
dtype02
gru_cell_9/ReadVariableOp�
gru_cell_9/unstackUnpack!gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2
gru_cell_9/unstack�
 gru_cell_9/MatMul/ReadVariableOpReadVariableOp)gru_cell_9_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02"
 gru_cell_9/MatMul/ReadVariableOp�
gru_cell_9/MatMulMatMulstrided_slice_2:output:0(gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
gru_cell_9/MatMul�
gru_cell_9/BiasAddBiasAddgru_cell_9/MatMul:product:0gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:����������2
gru_cell_9/BiasAddf
gru_cell_9/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_9/Const�
gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
gru_cell_9/split/split_dim�
gru_cell_9/splitSplit#gru_cell_9/split/split_dim:output:0gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
gru_cell_9/split�
"gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_9_matmul_1_readvariableop_resource*
_output_shapes
:	K�*
dtype02$
"gru_cell_9/MatMul_1/ReadVariableOp�
gru_cell_9/MatMul_1MatMulzeros:output:0*gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
gru_cell_9/MatMul_1�
gru_cell_9/BiasAdd_1BiasAddgru_cell_9/MatMul_1:product:0gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:����������2
gru_cell_9/BiasAdd_1}
gru_cell_9/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"K   K   ����2
gru_cell_9/Const_1�
gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
gru_cell_9/split_1/split_dim�
gru_cell_9/split_1SplitVgru_cell_9/BiasAdd_1:output:0gru_cell_9/Const_1:output:0%gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
gru_cell_9/split_1�
gru_cell_9/addAddV2gru_cell_9/split:output:0gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/addy
gru_cell_9/SigmoidSigmoidgru_cell_9/add:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/Sigmoid�
gru_cell_9/add_1AddV2gru_cell_9/split:output:1gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:���������K2
gru_cell_9/add_1
gru_cell_9/Sigmoid_1Sigmoidgru_cell_9/add_1:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/Sigmoid_1�
gru_cell_9/mulMulgru_cell_9/Sigmoid_1:y:0gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:���������K2
gru_cell_9/mul�
gru_cell_9/add_2AddV2gru_cell_9/split:output:2gru_cell_9/mul:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/add_2r
gru_cell_9/ReluRelugru_cell_9/add_2:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/Relu�
gru_cell_9/mul_1Mulgru_cell_9/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/mul_1i
gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
gru_cell_9/sub/x�
gru_cell_9/subSubgru_cell_9/sub/x:output:0gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/sub�
gru_cell_9/mul_2Mulgru_cell_9/sub:z:0gru_cell_9/Relu:activations:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/mul_2�
gru_cell_9/add_3AddV2gru_cell_9/mul_1:z:0gru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/add_3�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����K   2
TensorArrayV2_1/element_shape�
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
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_9_readvariableop_resource)gru_cell_9_matmul_readvariableop_resource+gru_cell_9_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������K: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_27550180*
condR
while_cond_27550179*8
output_shapes'
%: : : : :���������K: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����K   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������K*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
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
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������K*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������K2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime�
IdentityIdentitystrided_slice_3:output:0!^gru_cell_9/MatMul/ReadVariableOp#^gru_cell_9/MatMul_1/ReadVariableOp^gru_cell_9/ReadVariableOp^while*
T0*'
_output_shapes
:���������K2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::2D
 gru_cell_9/MatMul/ReadVariableOp gru_cell_9/MatMul/ReadVariableOp2H
"gru_cell_9/MatMul_1/ReadVariableOp"gru_cell_9/MatMul_1/ReadVariableOp26
gru_cell_9/ReadVariableOpgru_cell_9/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�	
�
F__inference_dense_19_layer_call_and_return_conditional_losses_27549422

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�	
�
gru_9_while_cond_27549810(
$gru_9_while_gru_9_while_loop_counter.
*gru_9_while_gru_9_while_maximum_iterations
gru_9_while_placeholder
gru_9_while_placeholder_1
gru_9_while_placeholder_2*
&gru_9_while_less_gru_9_strided_slice_1B
>gru_9_while_gru_9_while_cond_27549810___redundant_placeholder0B
>gru_9_while_gru_9_while_cond_27549810___redundant_placeholder1B
>gru_9_while_gru_9_while_cond_27549810___redundant_placeholder2B
>gru_9_while_gru_9_while_cond_27549810___redundant_placeholder3
gru_9_while_identity
�
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
-: : : : :���������K: ::::: 
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
:���������K:

_output_shapes
: :

_output_shapes
:
�
�
while_cond_27550179
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_27550179___redundant_placeholder06
2while_while_cond_27550179___redundant_placeholder16
2while_while_cond_27550179___redundant_placeholder26
2while_while_cond_27550179___redundant_placeholder3
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
-: : : : :���������K: ::::: 
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
:���������K:

_output_shapes
: :

_output_shapes
:
�F
�
!__inference__traced_save_27550898
file_prefix.
*savev2_dense_18_kernel_read_readvariableop,
(savev2_dense_18_bias_read_readvariableop.
*savev2_dense_19_kernel_read_readvariableop,
(savev2_dense_19_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop6
2savev2_gru_9_gru_cell_9_kernel_read_readvariableop@
<savev2_gru_9_gru_cell_9_recurrent_kernel_read_readvariableop4
0savev2_gru_9_gru_cell_9_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop5
1savev2_adam_dense_18_kernel_m_read_readvariableop3
/savev2_adam_dense_18_bias_m_read_readvariableop5
1savev2_adam_dense_19_kernel_m_read_readvariableop3
/savev2_adam_dense_19_bias_m_read_readvariableop=
9savev2_adam_gru_9_gru_cell_9_kernel_m_read_readvariableopG
Csavev2_adam_gru_9_gru_cell_9_recurrent_kernel_m_read_readvariableop;
7savev2_adam_gru_9_gru_cell_9_bias_m_read_readvariableop5
1savev2_adam_dense_18_kernel_v_read_readvariableop3
/savev2_adam_dense_18_bias_v_read_readvariableop5
1savev2_adam_dense_19_kernel_v_read_readvariableop3
/savev2_adam_dense_19_bias_v_read_readvariableop=
9savev2_adam_gru_9_gru_cell_9_kernel_v_read_readvariableopG
Csavev2_adam_gru_9_gru_cell_9_recurrent_kernel_v_read_readvariableop;
7savev2_adam_gru_9_gru_cell_9_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
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
Const_1�
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
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*�
value�B�!B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*U
valueLBJ!B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_18_kernel_read_readvariableop(savev2_dense_18_bias_read_readvariableop*savev2_dense_19_kernel_read_readvariableop(savev2_dense_19_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop2savev2_gru_9_gru_cell_9_kernel_read_readvariableop<savev2_gru_9_gru_cell_9_recurrent_kernel_read_readvariableop0savev2_gru_9_gru_cell_9_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop1savev2_adam_dense_18_kernel_m_read_readvariableop/savev2_adam_dense_18_bias_m_read_readvariableop1savev2_adam_dense_19_kernel_m_read_readvariableop/savev2_adam_dense_19_bias_m_read_readvariableop9savev2_adam_gru_9_gru_cell_9_kernel_m_read_readvariableopCsavev2_adam_gru_9_gru_cell_9_recurrent_kernel_m_read_readvariableop7savev2_adam_gru_9_gru_cell_9_bias_m_read_readvariableop1savev2_adam_dense_18_kernel_v_read_readvariableop/savev2_adam_dense_18_bias_v_read_readvariableop1savev2_adam_dense_19_kernel_v_read_readvariableop/savev2_adam_dense_19_bias_v_read_readvariableop9savev2_adam_gru_9_gru_cell_9_kernel_v_read_readvariableopCsavev2_adam_gru_9_gru_cell_9_recurrent_kernel_v_read_readvariableop7savev2_adam_gru_9_gru_cell_9_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 */
dtypes%
#2!	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
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

identity_1Identity_1:output:0*�
_input_shapes�
�: :Kd:d:d:: : : : : :	�:	K�:	�: : : : : : :Kd:d:d::	�:	K�:	�:Kd:d:d::	�:	K�:	�: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:Kd: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::
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
: :%
!

_output_shapes
:	�:%!

_output_shapes
:	K�:%!

_output_shapes
:	�:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:Kd: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::%!

_output_shapes
:	�:%!

_output_shapes
:	K�:%!

_output_shapes
:	�:$ 

_output_shapes

:Kd: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::%!

_output_shapes
:	�:%!

_output_shapes
:	K�:% !

_output_shapes
:	�:!

_output_shapes
: 
�<
�
C__inference_gru_9_layer_call_and_return_conditional_losses_27549025

inputs
gru_cell_9_27548949
gru_cell_9_27548951
gru_cell_9_27548953
identity��"gru_cell_9/StatefulPartitionedCall�whileD
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
strided_slice/stack_2�
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
B :�2
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
zeros/packed/1�
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
:���������K2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
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
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
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
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
"gru_cell_9/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_9_27548949gru_cell_9_27548951gru_cell_9_27548953*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������K:���������K*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_gru_cell_9_layer_call_and_return_conditional_losses_275485842$
"gru_cell_9/StatefulPartitionedCall�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����K   2
TensorArrayV2_1/element_shape�
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
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_9_27548949gru_cell_9_27548951gru_cell_9_27548953*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������K: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_27548961*
condR
while_cond_27548960*8
output_shapes'
%: : : : :���������K: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����K   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������K*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
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
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������K*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������K2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime�
IdentityIdentitystrided_slice_3:output:0#^gru_cell_9/StatefulPartitionedCall^while*
T0*'
_output_shapes
:���������K2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::2H
"gru_cell_9/StatefulPartitionedCall"gru_cell_9/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�Z
�
C__inference_gru_9_layer_call_and_return_conditional_losses_27550610

inputs&
"gru_cell_9_readvariableop_resource-
)gru_cell_9_matmul_readvariableop_resource/
+gru_cell_9_matmul_1_readvariableop_resource
identity�� gru_cell_9/MatMul/ReadVariableOp�"gru_cell_9/MatMul_1/ReadVariableOp�gru_cell_9/ReadVariableOp�whileD
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
strided_slice/stack_2�
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
B :�2
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
zeros/packed/1�
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
:���������K2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������2
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
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
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
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
gru_cell_9/ReadVariableOpReadVariableOp"gru_cell_9_readvariableop_resource*
_output_shapes
:	�*
dtype02
gru_cell_9/ReadVariableOp�
gru_cell_9/unstackUnpack!gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2
gru_cell_9/unstack�
 gru_cell_9/MatMul/ReadVariableOpReadVariableOp)gru_cell_9_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02"
 gru_cell_9/MatMul/ReadVariableOp�
gru_cell_9/MatMulMatMulstrided_slice_2:output:0(gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
gru_cell_9/MatMul�
gru_cell_9/BiasAddBiasAddgru_cell_9/MatMul:product:0gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:����������2
gru_cell_9/BiasAddf
gru_cell_9/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_9/Const�
gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
gru_cell_9/split/split_dim�
gru_cell_9/splitSplit#gru_cell_9/split/split_dim:output:0gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
gru_cell_9/split�
"gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_9_matmul_1_readvariableop_resource*
_output_shapes
:	K�*
dtype02$
"gru_cell_9/MatMul_1/ReadVariableOp�
gru_cell_9/MatMul_1MatMulzeros:output:0*gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
gru_cell_9/MatMul_1�
gru_cell_9/BiasAdd_1BiasAddgru_cell_9/MatMul_1:product:0gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:����������2
gru_cell_9/BiasAdd_1}
gru_cell_9/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"K   K   ����2
gru_cell_9/Const_1�
gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
gru_cell_9/split_1/split_dim�
gru_cell_9/split_1SplitVgru_cell_9/BiasAdd_1:output:0gru_cell_9/Const_1:output:0%gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
gru_cell_9/split_1�
gru_cell_9/addAddV2gru_cell_9/split:output:0gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/addy
gru_cell_9/SigmoidSigmoidgru_cell_9/add:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/Sigmoid�
gru_cell_9/add_1AddV2gru_cell_9/split:output:1gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:���������K2
gru_cell_9/add_1
gru_cell_9/Sigmoid_1Sigmoidgru_cell_9/add_1:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/Sigmoid_1�
gru_cell_9/mulMulgru_cell_9/Sigmoid_1:y:0gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:���������K2
gru_cell_9/mul�
gru_cell_9/add_2AddV2gru_cell_9/split:output:2gru_cell_9/mul:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/add_2r
gru_cell_9/ReluRelugru_cell_9/add_2:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/Relu�
gru_cell_9/mul_1Mulgru_cell_9/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/mul_1i
gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
gru_cell_9/sub/x�
gru_cell_9/subSubgru_cell_9/sub/x:output:0gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/sub�
gru_cell_9/mul_2Mulgru_cell_9/sub:z:0gru_cell_9/Relu:activations:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/mul_2�
gru_cell_9/add_3AddV2gru_cell_9/mul_1:z:0gru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:���������K2
gru_cell_9/add_3�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����K   2
TensorArrayV2_1/element_shape�
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
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_9_readvariableop_resource)gru_cell_9_matmul_readvariableop_resource+gru_cell_9_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������K: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_27550520*
condR
while_cond_27550519*8
output_shapes'
%: : : : :���������K: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����K   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������K*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
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
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������K*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������K2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime�
IdentityIdentitystrided_slice_3:output:0!^gru_cell_9/MatMul/ReadVariableOp#^gru_cell_9/MatMul_1/ReadVariableOp^gru_cell_9/ReadVariableOp^while*
T0*'
_output_shapes
:���������K2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������:::2D
 gru_cell_9/MatMul/ReadVariableOp gru_cell_9/MatMul/ReadVariableOp2H
"gru_cell_9/MatMul_1/ReadVariableOp"gru_cell_9/MatMul_1/ReadVariableOp26
gru_cell_9/ReadVariableOpgru_cell_9/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
F__inference_dense_19_layer_call_and_return_conditional_losses_27550662

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�F
�
while_body_27550361
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
1while_gru_cell_9_matmul_1_readvariableop_resource��&while/gru_cell_9/MatMul/ReadVariableOp�(while/gru_cell_9/MatMul_1/ReadVariableOp�while/gru_cell_9/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
while/gru_cell_9/ReadVariableOpReadVariableOp*while_gru_cell_9_readvariableop_resource_0*
_output_shapes
:	�*
dtype02!
while/gru_cell_9/ReadVariableOp�
while/gru_cell_9/unstackUnpack'while/gru_cell_9/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2
while/gru_cell_9/unstack�
&while/gru_cell_9/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02(
&while/gru_cell_9/MatMul/ReadVariableOp�
while/gru_cell_9/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/gru_cell_9/MatMul�
while/gru_cell_9/BiasAddBiasAdd!while/gru_cell_9/MatMul:product:0!while/gru_cell_9/unstack:output:0*
T0*(
_output_shapes
:����������2
while/gru_cell_9/BiasAddr
while/gru_cell_9/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_9/Const�
 while/gru_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2"
 while/gru_cell_9/split/split_dim�
while/gru_cell_9/splitSplit)while/gru_cell_9/split/split_dim:output:0!while/gru_cell_9/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
while/gru_cell_9/split�
(while/gru_cell_9/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes
:	K�*
dtype02*
(while/gru_cell_9/MatMul_1/ReadVariableOp�
while/gru_cell_9/MatMul_1MatMulwhile_placeholder_20while/gru_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/gru_cell_9/MatMul_1�
while/gru_cell_9/BiasAdd_1BiasAdd#while/gru_cell_9/MatMul_1:product:0!while/gru_cell_9/unstack:output:1*
T0*(
_output_shapes
:����������2
while/gru_cell_9/BiasAdd_1�
while/gru_cell_9/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"K   K   ����2
while/gru_cell_9/Const_1�
"while/gru_cell_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2$
"while/gru_cell_9/split_1/split_dim�
while/gru_cell_9/split_1SplitV#while/gru_cell_9/BiasAdd_1:output:0!while/gru_cell_9/Const_1:output:0+while/gru_cell_9/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
while/gru_cell_9/split_1�
while/gru_cell_9/addAddV2while/gru_cell_9/split:output:0!while/gru_cell_9/split_1:output:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/add�
while/gru_cell_9/SigmoidSigmoidwhile/gru_cell_9/add:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/Sigmoid�
while/gru_cell_9/add_1AddV2while/gru_cell_9/split:output:1!while/gru_cell_9/split_1:output:1*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/add_1�
while/gru_cell_9/Sigmoid_1Sigmoidwhile/gru_cell_9/add_1:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/Sigmoid_1�
while/gru_cell_9/mulMulwhile/gru_cell_9/Sigmoid_1:y:0!while/gru_cell_9/split_1:output:2*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/mul�
while/gru_cell_9/add_2AddV2while/gru_cell_9/split:output:2while/gru_cell_9/mul:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/add_2�
while/gru_cell_9/ReluReluwhile/gru_cell_9/add_2:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/Relu�
while/gru_cell_9/mul_1Mulwhile/gru_cell_9/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/mul_1u
while/gru_cell_9/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
while/gru_cell_9/sub/x�
while/gru_cell_9/subSubwhile/gru_cell_9/sub/x:output:0while/gru_cell_9/Sigmoid:y:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/sub�
while/gru_cell_9/mul_2Mulwhile/gru_cell_9/sub:z:0#while/gru_cell_9/Relu:activations:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/mul_2�
while/gru_cell_9/add_3AddV2while/gru_cell_9/mul_1:z:0while/gru_cell_9/mul_2:z:0*
T0*'
_output_shapes
:���������K2
while/gru_cell_9/add_3�
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
while/add_1�
while/IdentityIdentitywhile/add_1:z:0'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/gru_cell_9/add_3:z:0'^while/gru_cell_9/MatMul/ReadVariableOp)^while/gru_cell_9/MatMul_1/ReadVariableOp ^while/gru_cell_9/ReadVariableOp*
T0*'
_output_shapes
:���������K2
while/Identity_4"h
1while_gru_cell_9_matmul_1_readvariableop_resource3while_gru_cell_9_matmul_1_readvariableop_resource_0"d
/while_gru_cell_9_matmul_readvariableop_resource1while_gru_cell_9_matmul_readvariableop_resource_0"V
(while_gru_cell_9_readvariableop_resource*while_gru_cell_9_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :���������K: : :::2P
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
:���������K:

_output_shapes
: :

_output_shapes
: 
�
�
while_cond_27549105
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_27549105___redundant_placeholder06
2while_while_cond_27549105___redundant_placeholder16
2while_while_cond_27549105___redundant_placeholder26
2while_while_cond_27549105___redundant_placeholder3
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
-: : : : :���������K: ::::: 
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
:���������K:

_output_shapes
: :

_output_shapes
:
�
�
H__inference_gru_cell_9_layer_call_and_return_conditional_losses_27548544

inputs

states
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	�*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2	
unstack�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:����������2	
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
���������2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2
split�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	K�*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:����������2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"K   K   ����2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split_1/split_dim�
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������K:���������K:���������K*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:���������K2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������K2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:���������K2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������K2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:���������K2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:���������K2
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:���������K2
Relu\
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:���������K2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������K2
subd
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:���������K2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:���������K2
add_3�
IdentityIdentity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:���������K2

Identity�

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:���������K2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:���������:���������K:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������K
 
_user_specified_namestates
�	
�
-__inference_gru_cell_9_layer_call_fn_27550765

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������K:���������K*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_gru_cell_9_layer_call_and_return_conditional_losses_275485442
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������K2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������K2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:���������:���������K:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������K
"
_user_specified_name
states/0"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
G
gru_9_input8
serving_default_gru_9_input:0���������<
dense_190
StatefulPartitionedCall:0���������tensorflow/serving/predict:�
�*
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api
	
signatures
*a&call_and_return_all_conditional_losses
b_default_save_signature
c__call__"�(
_tf_keras_sequential�({"class_name": "Sequential", "name": "sequential_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "gru_9_input"}}, {"class_name": "GRU", "config": {"name": "gru_9", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 1]}, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 75, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}, {"class_name": "Dense", "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_19", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "gru_9_input"}}, {"class_name": "GRU", "config": {"name": "gru_9", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 1]}, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 75, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}, {"class_name": "Dense", "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_19", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mean_squared_error", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}}, {"class_name": "MeanMetricWrapper", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0010000000474974513, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�

cell

state_spec
trainable_variables
regularization_losses
	variables
	keras_api
*d&call_and_return_all_conditional_losses
e__call__"�
_tf_keras_rnn_layer�
{"class_name": "GRU", "name": "gru_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "gru_9", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 1]}, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 75, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 1]}}
�

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*f&call_and_return_all_conditional_losses
g__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 75}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 75]}}
�

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*h&call_and_return_all_conditional_losses
i__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_19", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
�
iter

beta_1

beta_2
	decay
 learning_ratemSmTmUmV!mW"mX#mYvZv[v\v]!v^"v_#v`"
	optimizer
Q
!0
"1
#2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
Q
!0
"1
#2
3
4
5
6"
trackable_list_wrapper
�
$layer_metrics

%layers
&layer_regularization_losses
trainable_variables
regularization_losses
'non_trainable_variables
(metrics
	variables
c__call__
b_default_save_signature
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
,
jserving_default"
signature_map
�

!kernel
"recurrent_kernel
#bias
)trainable_variables
*regularization_losses
+	variables
,	keras_api
*k&call_and_return_all_conditional_losses
l__call__"�
_tf_keras_layer�{"class_name": "GRUCell", "name": "gru_cell_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gru_cell_9", "trainable": true, "dtype": "float32", "units": 75, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}
 "
trackable_list_wrapper
5
!0
"1
#2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
!0
"1
#2"
trackable_list_wrapper
�
-layer_metrics

.layers
/layer_regularization_losses
trainable_variables
regularization_losses
0non_trainable_variables
1metrics

2states
	variables
e__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
!:Kd2dense_18/kernel
:d2dense_18/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
3layer_metrics

4layers
5layer_regularization_losses
trainable_variables
regularization_losses
6non_trainable_variables
7metrics
	variables
g__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
!:d2dense_19/kernel
:2dense_19/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
8layer_metrics

9layers
:layer_regularization_losses
trainable_variables
regularization_losses
;non_trainable_variables
<metrics
	variables
i__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
*:(	�2gru_9/gru_cell_9/kernel
4:2	K�2!gru_9/gru_cell_9/recurrent_kernel
(:&	�2gru_9/gru_cell_9/bias
 "
trackable_dict_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
=0
>1
?2"
trackable_list_wrapper
5
!0
"1
#2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
!0
"1
#2"
trackable_list_wrapper
�
@layer_metrics

Alayers
Blayer_regularization_losses
)trainable_variables
*regularization_losses
Cnon_trainable_variables
Dmetrics
+	variables
l__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
'

0"
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
�
	Etotal
	Fcount
G	variables
H	keras_api"�
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
�
	Itotal
	Jcount
K
_fn_kwargs
L	variables
M	keras_api"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "name": "mse", "dtype": "float32", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}}
�
	Ntotal
	Ocount
P
_fn_kwargs
Q	variables
R	keras_api"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "name": "mae", "dtype": "float32", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}}
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
.
E0
F1"
trackable_list_wrapper
-
G	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
I0
J1"
trackable_list_wrapper
-
L	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
N0
O1"
trackable_list_wrapper
-
Q	variables"
_generic_user_object
&:$Kd2Adam/dense_18/kernel/m
 :d2Adam/dense_18/bias/m
&:$d2Adam/dense_19/kernel/m
 :2Adam/dense_19/bias/m
/:-	�2Adam/gru_9/gru_cell_9/kernel/m
9:7	K�2(Adam/gru_9/gru_cell_9/recurrent_kernel/m
-:+	�2Adam/gru_9/gru_cell_9/bias/m
&:$Kd2Adam/dense_18/kernel/v
 :d2Adam/dense_18/bias/v
&:$d2Adam/dense_19/kernel/v
 :2Adam/dense_19/bias/v
/:-	�2Adam/gru_9/gru_cell_9/kernel/v
9:7	K�2(Adam/gru_9/gru_cell_9/recurrent_kernel/v
-:+	�2Adam/gru_9/gru_cell_9/bias/v
�2�
J__inference_sequential_9_layer_call_and_return_conditional_losses_27549460
J__inference_sequential_9_layer_call_and_return_conditional_losses_27549742
J__inference_sequential_9_layer_call_and_return_conditional_losses_27549914
J__inference_sequential_9_layer_call_and_return_conditional_losses_27549439�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
#__inference__wrapped_model_27548472�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *.�+
)�&
gru_9_input���������
�2�
/__inference_sequential_9_layer_call_fn_27549501
/__inference_sequential_9_layer_call_fn_27549541
/__inference_sequential_9_layer_call_fn_27549933
/__inference_sequential_9_layer_call_fn_27549952�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
C__inference_gru_9_layer_call_and_return_conditional_losses_27550111
C__inference_gru_9_layer_call_and_return_conditional_losses_27550270
C__inference_gru_9_layer_call_and_return_conditional_losses_27550451
C__inference_gru_9_layer_call_and_return_conditional_losses_27550610�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
(__inference_gru_9_layer_call_fn_27550281
(__inference_gru_9_layer_call_fn_27550621
(__inference_gru_9_layer_call_fn_27550632
(__inference_gru_9_layer_call_fn_27550292�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
F__inference_dense_18_layer_call_and_return_conditional_losses_27550643�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_dense_18_layer_call_fn_27550652�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_dense_19_layer_call_and_return_conditional_losses_27550662�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_dense_19_layer_call_fn_27550671�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
&__inference_signature_wrapper_27549570gru_9_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
H__inference_gru_cell_9_layer_call_and_return_conditional_losses_27550751
H__inference_gru_cell_9_layer_call_and_return_conditional_losses_27550711�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
-__inference_gru_cell_9_layer_call_fn_27550765
-__inference_gru_cell_9_layer_call_fn_27550779�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 �
#__inference__wrapped_model_27548472x#!"8�5
.�+
)�&
gru_9_input���������
� "3�0
.
dense_19"�
dense_19����������
F__inference_dense_18_layer_call_and_return_conditional_losses_27550643\/�,
%�"
 �
inputs���������K
� "%�"
�
0���������d
� ~
+__inference_dense_18_layer_call_fn_27550652O/�,
%�"
 �
inputs���������K
� "����������d�
F__inference_dense_19_layer_call_and_return_conditional_losses_27550662\/�,
%�"
 �
inputs���������d
� "%�"
�
0���������
� ~
+__inference_dense_19_layer_call_fn_27550671O/�,
%�"
 �
inputs���������d
� "�����������
C__inference_gru_9_layer_call_and_return_conditional_losses_27550111}#!"O�L
E�B
4�1
/�,
inputs/0������������������

 
p

 
� "%�"
�
0���������K
� �
C__inference_gru_9_layer_call_and_return_conditional_losses_27550270}#!"O�L
E�B
4�1
/�,
inputs/0������������������

 
p 

 
� "%�"
�
0���������K
� �
C__inference_gru_9_layer_call_and_return_conditional_losses_27550451m#!"?�<
5�2
$�!
inputs���������

 
p

 
� "%�"
�
0���������K
� �
C__inference_gru_9_layer_call_and_return_conditional_losses_27550610m#!"?�<
5�2
$�!
inputs���������

 
p 

 
� "%�"
�
0���������K
� �
(__inference_gru_9_layer_call_fn_27550281p#!"O�L
E�B
4�1
/�,
inputs/0������������������

 
p

 
� "����������K�
(__inference_gru_9_layer_call_fn_27550292p#!"O�L
E�B
4�1
/�,
inputs/0������������������

 
p 

 
� "����������K�
(__inference_gru_9_layer_call_fn_27550621`#!"?�<
5�2
$�!
inputs���������

 
p

 
� "����������K�
(__inference_gru_9_layer_call_fn_27550632`#!"?�<
5�2
$�!
inputs���������

 
p 

 
� "����������K�
H__inference_gru_cell_9_layer_call_and_return_conditional_losses_27550711�#!"\�Y
R�O
 �
inputs���������
'�$
"�
states/0���������K
p
� "R�O
H�E
�
0/0���������K
$�!
�
0/1/0���������K
� �
H__inference_gru_cell_9_layer_call_and_return_conditional_losses_27550751�#!"\�Y
R�O
 �
inputs���������
'�$
"�
states/0���������K
p 
� "R�O
H�E
�
0/0���������K
$�!
�
0/1/0���������K
� �
-__inference_gru_cell_9_layer_call_fn_27550765�#!"\�Y
R�O
 �
inputs���������
'�$
"�
states/0���������K
p
� "D�A
�
0���������K
"�
�
1/0���������K�
-__inference_gru_cell_9_layer_call_fn_27550779�#!"\�Y
R�O
 �
inputs���������
'�$
"�
states/0���������K
p 
� "D�A
�
0���������K
"�
�
1/0���������K�
J__inference_sequential_9_layer_call_and_return_conditional_losses_27549439r#!"@�=
6�3
)�&
gru_9_input���������
p

 
� "%�"
�
0���������
� �
J__inference_sequential_9_layer_call_and_return_conditional_losses_27549460r#!"@�=
6�3
)�&
gru_9_input���������
p 

 
� "%�"
�
0���������
� �
J__inference_sequential_9_layer_call_and_return_conditional_losses_27549742m#!";�8
1�.
$�!
inputs���������
p

 
� "%�"
�
0���������
� �
J__inference_sequential_9_layer_call_and_return_conditional_losses_27549914m#!";�8
1�.
$�!
inputs���������
p 

 
� "%�"
�
0���������
� �
/__inference_sequential_9_layer_call_fn_27549501e#!"@�=
6�3
)�&
gru_9_input���������
p

 
� "�����������
/__inference_sequential_9_layer_call_fn_27549541e#!"@�=
6�3
)�&
gru_9_input���������
p 

 
� "�����������
/__inference_sequential_9_layer_call_fn_27549933`#!";�8
1�.
$�!
inputs���������
p

 
� "�����������
/__inference_sequential_9_layer_call_fn_27549952`#!";�8
1�.
$�!
inputs���������
p 

 
� "�����������
&__inference_signature_wrapper_27549570�#!"G�D
� 
=�:
8
gru_9_input)�&
gru_9_input���������"3�0
.
dense_19"�
dense_19���������