from AddLora.peft.tuners.addmoelora import AddMOELoraLinear 

def set_runtime_task_id(model, task_id: int):
    for m in model.modules():
        # 你也可以换成 isinstance(m, AddMOELoraLinear)
        if isinstance(m, AddMOELoraLinear):
            setattr(m, "runtime_task_id", int(task_id)) 
        elif hasattr(m, "runtime_task_id"):
            m.runtime_task_id = int(task_id)