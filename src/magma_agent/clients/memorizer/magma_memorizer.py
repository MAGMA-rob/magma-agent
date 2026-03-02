from typing import Dict, List
import torch #type: ignore

from magma_agent.messages import BatchedMessageMemorizer
from .base import Memorizer

class MagmaMemorizer(Memorizer):

    def process_batched_entry(self, message : BatchedMessageMemorizer, inference_mode : bool) -> List[Dict]:
        memories = []
        
        for i in range(len(message.memory)):
            memory = "Memory:\n"
            for j, mem in enumerate(message.memory[i]):
                if j in message.preserved_memory_indices[i]:
                    id = "X"
                else:
                    id = str(j)
                memory += f"[{id}] {mem}\n"
            
            memories.append(memory)

        formatted_inputs = [
            self.tokenizer.apply_chat_template(
                [{}],
                think=message.think[i],
                memory=memories[i],
                tokenize=False,
                add_generation_prompt=True
            )
            for i in range(len(memories))
        ]

        inputs = self.tokenizer(formatted_inputs, return_tensors="pt", padding="max_length", max_length=900).to(self.model.device)
        input_lengths = [len(x) for x in inputs["input_ids"]]
        
        with torch.no_grad():
            if inference_mode:
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample = False,
                )
            else:
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.8,
                    top_p=0.95
                )

        responses = []
        for i in range(len(memories)):
            # print(message.memory[i])
            # print(message.preserved_memory_indices[i])
            generated_tokens = output[i][input_lengths[i]:]
            response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            # print(response_text)
            # print("----------")
            try:
                dic = self._transform_output_to_json(response_text)
                responses.append(dic)
            except:
                print(f"[MEMORIZER] Bad model output.")
                responses.append(response_text)

        return responses
    

    def _transform_output_to_json(self, data):
        r_id = []
        add_statement = []
        for line in data.splitlines():
            if not line.strip():
                continue  # skip empty lines

            cmd, args = line.split(maxsplit=1)

            if "REMOVE" in cmd:
                r_id.extend(args.split(','))

            elif "ADD" in cmd:
                add_statement.append(args)

        return [
            {
                "name" : "remove",
                "arguments" : {
                    "ids" : r_id
                }
            },
            {
                "name" : "add",
                "arguments" : {
                    "statements" : add_statement
                }
            }
        ]