־
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
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
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8��
l

value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
value/bias
e
value/bias/Read/ReadVariableOpReadVariableOp
value/bias*
_output_shapes
:*
dtype0
t
value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namevalue/kernel
m
 value/kernel/Read/ReadVariableOpReadVariableOpvalue/kernel*
_output_shapes

:@*
dtype0
n
action/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameaction/bias
g
action/bias/Read/ReadVariableOpReadVariableOpaction/bias*
_output_shapes
:*
dtype0
v
action/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_nameaction/kernel
o
!action/kernel/Read/ReadVariableOpReadVariableOpaction/kernel*
_output_shapes

:@*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:@*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:@*
dtype0
z
serving_default_input_1Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense/kernel
dense/biasvalue/kernel
value/biasaction/kernelaction/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������:���������*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8� *.
f)R'
%__inference_signature_wrapper_1407957

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

#kernel
$bias*
.
0
1
2
3
#4
$5*
.
0
1
2
3
#4
$5*
	
%0* 
�
&non_trainable_variables

'layers
(metrics
)layer_regularization_losses
*layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*
6
+trace_0
,trace_1
-trace_2
.trace_3* 
6
/trace_0
0trace_1
1trace_2
2trace_3* 
* 

3serving_default* 

0
1*

0
1*
	
%0* 
�
4non_trainable_variables

5layers
6metrics
7layer_regularization_losses
8layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

9trace_0* 

:trace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
�
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

@trace_0* 

Atrace_0* 
]W
VARIABLE_VALUEaction/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEaction/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

#0
$1*

#0
$1*
* 
�
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*

Gtrace_0* 

Htrace_0* 
\V
VARIABLE_VALUEvalue/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
value/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

Itrace_0* 
* 
 
0
1
2
3*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
%0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp!action/kernel/Read/ReadVariableOpaction/bias/Read/ReadVariableOp value/kernel/Read/ReadVariableOpvalue/bias/Read/ReadVariableOpConst*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *)
f$R"
 __inference__traced_save_1408171
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasaction/kernelaction/biasvalue/kernel
value/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *,
f'R%
#__inference__traced_restore_1408199��
�
�
"__inference__wrapped_model_1407689
input_1<
*model_dense_matmul_readvariableop_resource:@9
+model_dense_biasadd_readvariableop_resource:@<
*model_value_matmul_readvariableop_resource:@9
+model_value_biasadd_readvariableop_resource:=
+model_action_matmul_readvariableop_resource:@:
,model_action_biasadd_readvariableop_resource:
identity

identity_1��#model/action/BiasAdd/ReadVariableOp�"model/action/MatMul/ReadVariableOp�"model/dense/BiasAdd/ReadVariableOp�!model/dense/MatMul/ReadVariableOp�"model/value/BiasAdd/ReadVariableOp�!model/value/MatMul/ReadVariableOp�
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
model/dense/MatMulMatMulinput_1)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@h
model/dense/ReluRelumodel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
!model/value/MatMul/ReadVariableOpReadVariableOp*model_value_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
model/value/MatMulMatMulmodel/dense/Relu:activations:0)model/value/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
"model/value/BiasAdd/ReadVariableOpReadVariableOp+model_value_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/value/BiasAddBiasAddmodel/value/MatMul:product:0*model/value/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
"model/action/MatMul/ReadVariableOpReadVariableOp+model_action_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
model/action/MatMulMatMulmodel/dense/Relu:activations:0*model/action/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
#model/action/BiasAdd/ReadVariableOpReadVariableOp,model_action_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/action/BiasAddBiasAddmodel/action/MatMul:product:0+model/action/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������p
model/action/SoftmaxSoftmaxmodel/action/BiasAdd:output:0*
T0*'
_output_shapes
:���������m
IdentityIdentitymodel/action/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������m

Identity_1Identitymodel/value/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp$^model/action/BiasAdd/ReadVariableOp#^model/action/MatMul/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp#^model/value/BiasAdd/ReadVariableOp"^model/value/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2J
#model/action/BiasAdd/ReadVariableOp#model/action/BiasAdd/ReadVariableOp2H
"model/action/MatMul/ReadVariableOp"model/action/MatMul/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2H
"model/value/BiasAdd/ReadVariableOp"model/value/BiasAdd/ReadVariableOp2F
!model/value/MatMul/ReadVariableOp!model/value/MatMul/ReadVariableOp:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�!
�
B__inference_model_layer_call_and_return_conditional_losses_1408057

inputs6
$dense_matmul_readvariableop_resource:@3
%dense_biasadd_readvariableop_resource:@6
$value_matmul_readvariableop_resource:@3
%value_biasadd_readvariableop_resource:7
%action_matmul_readvariableop_resource:@4
&action_biasadd_readvariableop_resource:
identity

identity_1��action/BiasAdd/ReadVariableOp�action/MatMul/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�.dense/kernel/Regularizer/L2Loss/ReadVariableOp�value/BiasAdd/ReadVariableOp�value/MatMul/ReadVariableOp�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0u
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
value/MatMul/ReadVariableOpReadVariableOp$value_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
value/MatMulMatMuldense/Relu:activations:0#value/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������~
value/BiasAdd/ReadVariableOpReadVariableOp%value_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
value/BiasAddBiasAddvalue/MatMul:product:0$value/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
action/MatMul/ReadVariableOpReadVariableOp%action_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
action/MatMulMatMuldense/Relu:activations:0$action/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
action/BiasAdd/ReadVariableOpReadVariableOp&action_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
action/BiasAddBiasAddaction/MatMul:product:0%action/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d
action/SoftmaxSoftmaxaction/BiasAdd:output:0*
T0*'
_output_shapes
:����������
.dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense/kernel/Regularizer/L2LossL2Loss6dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0(dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: g
IdentityIdentityaction/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������g

Identity_1Identityvalue/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^action/BiasAdd/ReadVariableOp^action/MatMul/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp/^dense/kernel/Regularizer/L2Loss/ReadVariableOp^value/BiasAdd/ReadVariableOp^value/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2>
action/BiasAdd/ReadVariableOpaction/BiasAdd/ReadVariableOp2<
action/MatMul/ReadVariableOpaction/MatMul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2`
.dense/kernel/Regularizer/L2Loss/ReadVariableOp.dense/kernel/Regularizer/L2Loss/ReadVariableOp2<
value/BiasAdd/ReadVariableOpvalue/BiasAdd/ReadVariableOp2:
value/MatMul/ReadVariableOpvalue/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_0_1408129I
7dense_kernel_regularizer_l2loss_readvariableop_resource:@
identity��.dense/kernel/Regularizer/L2Loss/ReadVariableOp�
.dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp7dense_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense/kernel/Regularizer/L2LossL2Loss6dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0(dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ^
IdentityIdentity dense/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: w
NoOpNoOp/^dense/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.dense/kernel/Regularizer/L2Loss/ReadVariableOp.dense/kernel/Regularizer/L2Loss/ReadVariableOp
�
�
(__inference_action_layer_call_fn_1408090

inputs
unknown:@
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *L
fGRE
C__inference_action_layer_call_and_return_conditional_losses_1407744o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
B__inference_dense_layer_call_and_return_conditional_losses_1407711

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�.dense/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@�
.dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense/kernel/Regularizer/L2LossL2Loss6dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0(dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense/kernel/Regularizer/L2Loss/ReadVariableOp.dense/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
C__inference_action_layer_call_and_return_conditional_losses_1407744

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
B__inference_value_layer_call_and_return_conditional_losses_1408120

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
%__inference_signature_wrapper_1407957
input_1
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:
	unknown_3:@
	unknown_4:
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������:���������*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8� *+
f&R$
"__inference__wrapped_model_1407689o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�

�
'__inference_model_layer_call_fn_1407980

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:
	unknown_3:@
	unknown_4:
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������:���������*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_1407756o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
B__inference_model_layer_call_and_return_conditional_losses_1407848

inputs
dense_1407827:@
dense_1407829:@
value_1407832:@
value_1407834: 
action_1407837:@
action_1407839:
identity

identity_1��action/StatefulPartitionedCall�dense/StatefulPartitionedCall�.dense/kernel/Regularizer/L2Loss/ReadVariableOp�value/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_1407827dense_1407829*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1407711�
value/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0value_1407832value_1407834*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_value_layer_call_and_return_conditional_losses_1407727�
action/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0action_1407837action_1407839*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *L
fGRE
C__inference_action_layer_call_and_return_conditional_losses_1407744|
.dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1407827*
_output_shapes

:@*
dtype0�
dense/kernel/Regularizer/L2LossL2Loss6dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0(dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: v
IdentityIdentity'action/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������w

Identity_1Identity&value/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^action/StatefulPartitionedCall^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/L2Loss/ReadVariableOp^value/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2@
action/StatefulPartitionedCallaction/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2`
.dense/kernel/Regularizer/L2Loss/ReadVariableOp.dense/kernel/Regularizer/L2Loss/ReadVariableOp2>
value/StatefulPartitionedCallvalue/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
B__inference_dense_layer_call_and_return_conditional_losses_1408081

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�.dense/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@�
.dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense/kernel/Regularizer/L2LossL2Loss6dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0(dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense/kernel/Regularizer/L2Loss/ReadVariableOp.dense/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
 __inference__traced_save_1408171
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop,
(savev2_action_kernel_read_readvariableop*
&savev2_action_bias_read_readvariableop+
'savev2_value_kernel_read_readvariableop)
%savev2_value_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
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
_temp/part�
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
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH{
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop(savev2_action_kernel_read_readvariableop&savev2_action_bias_read_readvariableop'savev2_value_kernel_read_readvariableop%savev2_value_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
	2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
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

identity_1Identity_1:output:0*G
_input_shapes6
4: :@:@:@::@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: 
�
�
B__inference_model_layer_call_and_return_conditional_losses_1407932
input_1
dense_1407911:@
dense_1407913:@
value_1407916:@
value_1407918: 
action_1407921:@
action_1407923:
identity

identity_1��action/StatefulPartitionedCall�dense/StatefulPartitionedCall�.dense/kernel/Regularizer/L2Loss/ReadVariableOp�value/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_1407911dense_1407913*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1407711�
value/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0value_1407916value_1407918*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_value_layer_call_and_return_conditional_losses_1407727�
action/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0action_1407921action_1407923*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *L
fGRE
C__inference_action_layer_call_and_return_conditional_losses_1407744|
.dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1407911*
_output_shapes

:@*
dtype0�
dense/kernel/Regularizer/L2LossL2Loss6dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0(dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: v
IdentityIdentity'action/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������w

Identity_1Identity&value/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^action/StatefulPartitionedCall^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/L2Loss/ReadVariableOp^value/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2@
action/StatefulPartitionedCallaction/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2`
.dense/kernel/Regularizer/L2Loss/ReadVariableOp.dense/kernel/Regularizer/L2Loss/ReadVariableOp2>
value/StatefulPartitionedCallvalue/StatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�

�
'__inference_model_layer_call_fn_1407884
input_1
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:
	unknown_3:@
	unknown_4:
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������:���������*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_1407848o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�

�
C__inference_action_layer_call_and_return_conditional_losses_1408101

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
B__inference_model_layer_call_and_return_conditional_losses_1407908
input_1
dense_1407887:@
dense_1407889:@
value_1407892:@
value_1407894: 
action_1407897:@
action_1407899:
identity

identity_1��action/StatefulPartitionedCall�dense/StatefulPartitionedCall�.dense/kernel/Regularizer/L2Loss/ReadVariableOp�value/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_1407887dense_1407889*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1407711�
value/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0value_1407892value_1407894*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_value_layer_call_and_return_conditional_losses_1407727�
action/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0action_1407897action_1407899*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *L
fGRE
C__inference_action_layer_call_and_return_conditional_losses_1407744|
.dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1407887*
_output_shapes

:@*
dtype0�
dense/kernel/Regularizer/L2LossL2Loss6dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0(dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: v
IdentityIdentity'action/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������w

Identity_1Identity&value/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^action/StatefulPartitionedCall^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/L2Loss/ReadVariableOp^value/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2@
action/StatefulPartitionedCallaction/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2`
.dense/kernel/Regularizer/L2Loss/ReadVariableOp.dense/kernel/Regularizer/L2Loss/ReadVariableOp2>
value/StatefulPartitionedCallvalue/StatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�	
�
B__inference_value_layer_call_and_return_conditional_losses_1407727

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
#__inference__traced_restore_1408199
file_prefix/
assignvariableop_dense_kernel:@+
assignvariableop_1_dense_bias:@2
 assignvariableop_2_action_kernel:@,
assignvariableop_3_action_bias:1
assignvariableop_4_value_kernel:@+
assignvariableop_5_value_bias:

identity_7��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH~
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp assignvariableop_2_action_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_action_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_value_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_value_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_7IdentityIdentity_6:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5*"
_acd_function_control_output(*
_output_shapes
 "!

identity_7Identity_7:output:0*!
_input_shapes
: : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_5:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�

�
'__inference_model_layer_call_fn_1407773
input_1
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:
	unknown_3:@
	unknown_4:
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������:���������*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_1407756o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
'__inference_dense_layer_call_fn_1408066

inputs
unknown:@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1407711o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
'__inference_value_layer_call_fn_1408110

inputs
unknown:@
	unknown_0:
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
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_value_layer_call_and_return_conditional_losses_1407727o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�!
�
B__inference_model_layer_call_and_return_conditional_losses_1408028

inputs6
$dense_matmul_readvariableop_resource:@3
%dense_biasadd_readvariableop_resource:@6
$value_matmul_readvariableop_resource:@3
%value_biasadd_readvariableop_resource:7
%action_matmul_readvariableop_resource:@4
&action_biasadd_readvariableop_resource:
identity

identity_1��action/BiasAdd/ReadVariableOp�action/MatMul/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�.dense/kernel/Regularizer/L2Loss/ReadVariableOp�value/BiasAdd/ReadVariableOp�value/MatMul/ReadVariableOp�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0u
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
value/MatMul/ReadVariableOpReadVariableOp$value_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
value/MatMulMatMuldense/Relu:activations:0#value/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������~
value/BiasAdd/ReadVariableOpReadVariableOp%value_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
value/BiasAddBiasAddvalue/MatMul:product:0$value/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
action/MatMul/ReadVariableOpReadVariableOp%action_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
action/MatMulMatMuldense/Relu:activations:0$action/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
action/BiasAdd/ReadVariableOpReadVariableOp&action_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
action/BiasAddBiasAddaction/MatMul:product:0%action/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d
action/SoftmaxSoftmaxaction/BiasAdd:output:0*
T0*'
_output_shapes
:����������
.dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense/kernel/Regularizer/L2LossL2Loss6dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0(dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: g
IdentityIdentityaction/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������g

Identity_1Identityvalue/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^action/BiasAdd/ReadVariableOp^action/MatMul/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp/^dense/kernel/Regularizer/L2Loss/ReadVariableOp^value/BiasAdd/ReadVariableOp^value/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2>
action/BiasAdd/ReadVariableOpaction/BiasAdd/ReadVariableOp2<
action/MatMul/ReadVariableOpaction/MatMul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2`
.dense/kernel/Regularizer/L2Loss/ReadVariableOp.dense/kernel/Regularizer/L2Loss/ReadVariableOp2<
value/BiasAdd/ReadVariableOpvalue/BiasAdd/ReadVariableOp2:
value/MatMul/ReadVariableOpvalue/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
'__inference_model_layer_call_fn_1407999

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:
	unknown_3:@
	unknown_4:
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������:���������*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_1407848o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
B__inference_model_layer_call_and_return_conditional_losses_1407756

inputs
dense_1407712:@
dense_1407714:@
value_1407728:@
value_1407730: 
action_1407745:@
action_1407747:
identity

identity_1��action/StatefulPartitionedCall�dense/StatefulPartitionedCall�.dense/kernel/Regularizer/L2Loss/ReadVariableOp�value/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_1407712dense_1407714*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1407711�
value/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0value_1407728value_1407730*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_value_layer_call_and_return_conditional_losses_1407727�
action/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0action_1407745action_1407747*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *L
fGRE
C__inference_action_layer_call_and_return_conditional_losses_1407744|
.dense/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_1407712*
_output_shapes

:@*
dtype0�
dense/kernel/Regularizer/L2LossL2Loss6dense/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0(dense/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: v
IdentityIdentity'action/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������w

Identity_1Identity&value/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^action/StatefulPartitionedCall^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/L2Loss/ReadVariableOp^value/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2@
action/StatefulPartitionedCallaction/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2`
.dense/kernel/Regularizer/L2Loss/ReadVariableOp.dense/kernel/Regularizer/L2Loss/ReadVariableOp2>
value/StatefulPartitionedCallvalue/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs"�	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
;
input_10
serving_default_input_1:0���������:
action0
StatefulPartitionedCall:0���������9
value0
StatefulPartitionedCall:1���������tensorflow/serving/predict:�i
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

#kernel
$bias"
_tf_keras_layer
J
0
1
2
3
#4
$5"
trackable_list_wrapper
J
0
1
2
3
#4
$5"
trackable_list_wrapper
'
%0"
trackable_list_wrapper
�
&non_trainable_variables

'layers
(metrics
)layer_regularization_losses
*layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
�
+trace_0
,trace_1
-trace_2
.trace_32�
'__inference_model_layer_call_fn_1407773
'__inference_model_layer_call_fn_1407980
'__inference_model_layer_call_fn_1407999
'__inference_model_layer_call_fn_1407884�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z+trace_0z,trace_1z-trace_2z.trace_3
�
/trace_0
0trace_1
1trace_2
2trace_32�
B__inference_model_layer_call_and_return_conditional_losses_1408028
B__inference_model_layer_call_and_return_conditional_losses_1408057
B__inference_model_layer_call_and_return_conditional_losses_1407908
B__inference_model_layer_call_and_return_conditional_losses_1407932�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z/trace_0z0trace_1z1trace_2z2trace_3
�B�
"__inference__wrapped_model_1407689input_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
,
3serving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
%0"
trackable_list_wrapper
�
4non_trainable_variables

5layers
6metrics
7layer_regularization_losses
8layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
9trace_02�
'__inference_dense_layer_call_fn_1408066�
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
 z9trace_0
�
:trace_02�
B__inference_dense_layer_call_and_return_conditional_losses_1408081�
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
 z:trace_0
:@2dense/kernel
:@2
dense/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
@trace_02�
(__inference_action_layer_call_fn_1408090�
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
 z@trace_0
�
Atrace_02�
C__inference_action_layer_call_and_return_conditional_losses_1408101�
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
 zAtrace_0
:@2action/kernel
:2action/bias
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
�
Gtrace_02�
'__inference_value_layer_call_fn_1408110�
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
 zGtrace_0
�
Htrace_02�
B__inference_value_layer_call_and_return_conditional_losses_1408120�
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
 zHtrace_0
:@2value/kernel
:2
value/bias
�
Itrace_02�
__inference_loss_fn_0_1408129�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� zItrace_0
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_model_layer_call_fn_1407773input_1"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
'__inference_model_layer_call_fn_1407980inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
'__inference_model_layer_call_fn_1407999inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
'__inference_model_layer_call_fn_1407884input_1"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_model_layer_call_and_return_conditional_losses_1408028inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_model_layer_call_and_return_conditional_losses_1408057inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_model_layer_call_and_return_conditional_losses_1407908input_1"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_model_layer_call_and_return_conditional_losses_1407932input_1"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_signature_wrapper_1407957input_1"�
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
%0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_dense_layer_call_fn_1408066inputs"�
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
B__inference_dense_layer_call_and_return_conditional_losses_1408081inputs"�
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
�B�
(__inference_action_layer_call_fn_1408090inputs"�
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
C__inference_action_layer_call_and_return_conditional_losses_1408101inputs"�
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
�B�
'__inference_value_layer_call_fn_1408110inputs"�
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
B__inference_value_layer_call_and_return_conditional_losses_1408120inputs"�
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
__inference_loss_fn_0_1408129"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� �
"__inference__wrapped_model_1407689�#$0�-
&�#
!�
input_1���������
� "Y�V
*
action �
action���������
(
value�
value����������
C__inference_action_layer_call_and_return_conditional_losses_1408101\/�,
%�"
 �
inputs���������@
� "%�"
�
0���������
� {
(__inference_action_layer_call_fn_1408090O/�,
%�"
 �
inputs���������@
� "�����������
B__inference_dense_layer_call_and_return_conditional_losses_1408081\/�,
%�"
 �
inputs���������
� "%�"
�
0���������@
� z
'__inference_dense_layer_call_fn_1408066O/�,
%�"
 �
inputs���������
� "����������@<
__inference_loss_fn_0_1408129�

� 
� "� �
B__inference_model_layer_call_and_return_conditional_losses_1407908�#$8�5
.�+
!�
input_1���������
p 

 
� "K�H
A�>
�
0/0���������
�
0/1���������
� �
B__inference_model_layer_call_and_return_conditional_losses_1407932�#$8�5
.�+
!�
input_1���������
p

 
� "K�H
A�>
�
0/0���������
�
0/1���������
� �
B__inference_model_layer_call_and_return_conditional_losses_1408028�#$7�4
-�*
 �
inputs���������
p 

 
� "K�H
A�>
�
0/0���������
�
0/1���������
� �
B__inference_model_layer_call_and_return_conditional_losses_1408057�#$7�4
-�*
 �
inputs���������
p

 
� "K�H
A�>
�
0/0���������
�
0/1���������
� �
'__inference_model_layer_call_fn_1407773�#$8�5
.�+
!�
input_1���������
p 

 
� "=�:
�
0���������
�
1����������
'__inference_model_layer_call_fn_1407884�#$8�5
.�+
!�
input_1���������
p

 
� "=�:
�
0���������
�
1����������
'__inference_model_layer_call_fn_1407980�#$7�4
-�*
 �
inputs���������
p 

 
� "=�:
�
0���������
�
1����������
'__inference_model_layer_call_fn_1407999�#$7�4
-�*
 �
inputs���������
p

 
� "=�:
�
0���������
�
1����������
%__inference_signature_wrapper_1407957�#$;�8
� 
1�.
,
input_1!�
input_1���������"Y�V
*
action �
action���������
(
value�
value����������
B__inference_value_layer_call_and_return_conditional_losses_1408120\#$/�,
%�"
 �
inputs���������@
� "%�"
�
0���������
� z
'__inference_value_layer_call_fn_1408110O#$/�,
%�"
 �
inputs���������@
� "����������