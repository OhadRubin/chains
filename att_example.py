import sys
import os
import random
from typing import List, Optional
from pydantic import BaseModel, Field
from dataclasses import dataclass
from src.prompt_chain import PromptChain, Prompt
from src.msg_chain import MessageChain


# Define models
class Attribute(BaseModel):
    name: str = Field(..., description="Name of the attribute")
    description: str = Field(..., description="Description of the attribute")
    importance_rank: int = Field(
        None, description="Numerical ranking of importance, lower is more important"
    )

    def __str__(self) -> str:
        return f"{self.name}: {self.description}"


class AttributeList(BaseModel):
    attributes: List[Attribute] = Field(..., description="List of attributes")

    def att_into_str(self, should_shuffle: bool = False, top_n: int = None) -> str:
        attributes = sorted(self.attributes, key=lambda x: x.importance_rank)
        if top_n is not None:
            attributes = attributes[:top_n]
        if should_shuffle:
            random.seed(42)
            random.shuffle(attributes)
        return "\n".join(str(attribute) for attribute in attributes)

    def sorted_att(self):
        return sorted(self.attributes, key=lambda x: x.importance_rank)


class QualityAttribute(BaseModel):
    name: str = Field(..., description="Name of the quality attribute")
    completion_percentage: int = Field(
        ..., description="The percentage of completion (0-100)"
    )


class DevelopmentStage(BaseModel):
    name: str = Field(..., description="Name of the development stage")
    description: str = Field(..., description="Description of the stage")
    state: List[QualityAttribute] = Field(
        ..., description="State of quality attributes"
    )


class DevModel(BaseModel):
    stages: List[DevelopmentStage]

    def as_str(self):
        result = []
        previous_attributes = {}

        for stage in self.stages:
            result.append(f"Stage: {stage.name}")
            result.append("Attributes:")

            result.append("\nChanges from previous stage:")
            for attribute in stage.state:
                prev_value = previous_attributes.get(attribute.name, 0)
                delta = attribute.completion_percentage - prev_value
                if delta != 0:
                    result.append(
                        f"  - {attribute.name}: {'+' if delta > 0 else ''}{delta}%"
                    )

            previous_attributes = {
                attr.name: attr.completion_percentage for attr in stage.state
            }

            result.append("-" * 50)
            result.append("")

        return "\n".join(result)


class DevelopmentStageText(BaseModel):
    name: str = Field(..., description="Name of the development stage")
    description: str = Field(..., description="Description of the stage")
    state_change: str = Field(
        ...,
        description="Description of how the state of our goal changed from the previous stage as described by the attributes.",
    )


class DevModelText(BaseModel):
    stages: List[DevelopmentStageText]


# Constants and templates
GOAL_POWER_LAW = """# Generalized Goal Achievement Power-Law

Progressing through successive stages of any complex goal typically involves increasingly difficult, resource-intensive, and risky transitions. These incremental difficulties often follow a **power-law distribution**, meaning that each subsequent step toward higher completion levels generally requires exponentially greater investments in terms of effort, time, and resources.

This relationship is widely observed across various domains:

- **Initial conception (0-10%)**: Brainstorming, ideation, and basic planning require minimal resources, primarily intellectual effort and time for conceptualization.

- **Foundation building (11-25%)**: Establishing fundamental structures, gathering essential resources, and creating basic frameworks demand modest but increased investment.

- **Early development (26-40%)**: Building core components, demonstrating basic functionality, and validating initial assumptions require noticeable increases in resource allocation.

- **Mid-stage development (41-60%)**: Integration of components, resolving technical challenges, and validating the approach substantially escalate complexity and resource requirements.

- **Advanced development (61-75%)**: System optimization, addressing edge cases, and preparing for completion significantly increase demands on expertise and resources.

- **Refinement (76-90%)**: Polishing, extensive testing, and ensuring reliability often require disproportionately large investments compared to earlier stages.

- **Final completion (91-100%)**: Achieving the highest levels of performance, reliability, and readiness typically demands the most intensive resource allocation, frequently exhibiting exponential growth in costs and efforts.

Goal progression tends to follow a **power-law pattern** across these stages, with each step significantly costlier and more challenging than the previous, regardless of the specific domain."""

ATT_DESC = "For example: Clarity, Completeness, Correctness etc.\nYou should rank the most important attributes higher (i.e a lower number)."


def combin_attr(chain):
    attributes = AttributeList(
        attributes=list(chain.response_list[-2].sorted_att())
        + chain.response_list[-1].attributes
    )
    return {"attributes_str": attributes.att_into_str(should_shuffle=True)}


# Set the target goal to analyze


def generate_attributes(target_goal: str):
    chain = (
    PromptChain().prompt(
        """What would you say are the attributes of a "{target_goal}" from a quality perspective? 
{ATT_DESC}
Give 30 such attributes.""",
        ).set_prev_fields({"target_goal": target_goal, "ATT_DESC": ATT_DESC})
        .set_model(lambda: MessageChain.get_chain(model="instructor"))
        .with_structure(AttributeList)
        .generate()
        .post_last(attribute_str=lambda x: x.att_into_str(should_shuffle=True))
        .gen_prompt(
        """Please re-rank the following attributes that define the quality of a "{target_goal}".
{ATT_DESC}
The attributes are:
{attribute_str}"""
        )
        .post_last(attribute_str=lambda x: x.att_into_str(top_n=5, should_shuffle=True))
        .gen_prompt(
        """Please suggest additional 5 attributes that define the quality of a "{target_goal}".
{ATT_DESC}
Do not suggest attributes that are already in the list.
The attributes are:
{attribute_str}"""
        )
        .post_chain(combin_attr)
        .gen_prompt(
        """Please re-rank the following attributes that define the quality of a "{target_goal}".
{ATT_DESC}
The attributes are:
{attribute_str}"""
        )
        .post_last(attribute_str=lambda x: x.att_into_str(top_n=10, should_shuffle=True))
        .gen_prompt(
        """Please merge the following attributes such that they are non-overlapping.
You should only merge attributes that are similar. Do not try and coerce two unrelated attributes into one.
Do: Testability and Verifiability.
Don't: Testability and Completeness.
Don't: Clarity and Precision.
The attributes define the quality of a "{target_goal}".
{ATT_DESC}
The attributes are:
{attribute_str}
"""
        )
        .post_last(attribute_str=lambda x: x.att_into_str(should_shuffle=False))
        .gen_prompt(
        """Please suggest 10 additional attributes that define the quality of a "{target_goal}".
{ATT_DESC}
The current attributes I have are:
{attribute_str}"""
        )
        .post_chain(combin_attr)
        .gen_prompt(
        """Please re-rank the following attributes that define the quality of a "{target_goal}".
{ATT_DESC}
The attributes are:
{attributes_str}"""
        )
        .post_last(attribute_str=lambda x: x.att_into_str(should_shuffle=True, top_n=10))
        .gen_prompt(
        """Please merge the following attributes such that they are non-overlapping. 
You should only merge attributes that are similar. Do not try and coerce two unrelated attributes into one.
Do: Testability and Verifiability.
Don't: Testability and Completeness.
Don't: Clarity and Precision.
The attributes define the quality of a "{target_goal}".
{ATT_DESC}
The attributes are:
{attributes_str}"""
        )
        .generate()
    )
    return chain.response_list[-1]




def get_development_stages(target_goal, attributes):

    chain = (
        PromptChain()
        .prompt(
            """We are interested in defining the different stages of developing a "{target_goal}" using the given quality attributes.

```
{GOAL_POWER_LAW}
```

These are the attributes of an "{target_goal}" from a quality perspective:
{attributes_str}

Suggest 10 stages for developing a "{target_goal}". It should be represented by the following schema and it should use the above attributes of a "{target_goal}" from a quality perspective.
It should also incorporate the following facts:
1. Attributes that not as important should be not be advance at all in the starting stages, and should be done mostly at later stages.
2. Progress should follow a power-law pattern.
3. Progress is not required to be monotonic,
4. It makes sense for most attributes not to increase at all in some stages.
5. The least important attributes should only be advanced at the later stages.


i.e we might be increase some attribute A at a previous stage, but at this stage it was decreased, and some attribute B was increased.

Refrain from adjactives as much as possible."""
        )
        .set_prev_fields({"target_goal": target_goal, 
                          "attributes_str": attributes.att_into_str(should_shuffle=False),
                          "GOAL_POWER_LAW": GOAL_POWER_LAW})
        .set_model(lambda: MessageChain.get_chain(model="instructor"))
        .with_structure(DevModel)
        .generate()
    )

    return chain.response_list[-1]




def get_development_text(target_goal, attributes, dev_model):
    attributes_str = attributes.att_into_str(should_shuffle=False)
    stages_str = dev_model.as_str()

    chain = (
        PromptChain()
        .prompt("""We are interested in defining the different stages of developing a "{target_goal}" with an emphesis on how quality w.r.t specific attributes evolves.
Meaning, at a specific stage of development, we can focus on a specific attribute A, and at another we can focus on another attribute B.
You will be given a list of attributes that determine the quality of a "{target_goal}", and a list of stages that changed from the previous stage and measure the quality of the "{target_goal}" with respect to the attributes.

Your task is to describe the changes from each stage to the next in words in such a way that it is clear to somewhere what he should focus on in each stage.
Your description should be in words and not in numbers and describe what should a person focus on during each stage.
For example, in early stages, development might prioritize certain attributes, while later stages could shift focus to different attributes as the project matures.
You should describe it according to how the attribute changes from the previous stage.
It should also be clear what is the goal of the stage.

List of attributes:
<attributes>
{attributes_str}
</attributes>

List of stages:
<stages>
{stages_str}
</stages>


Your task is to describe the changes from each stage to the next in words in such a way that it is clear to somewhere what he should focus on in each stage.
Your description should be in words and not in numbers and describe what should a person focus on during each stage.
For example, in early stages, development might prioritize certain attributes, while later stages could shift focus to different attributes as the project matures.
You should describe it according to how the attribute changes from the previous stage.
It should also be clear what is the goal of the stage.
Refrain from adjactives as much as possible.""")
        .set_prev_fields(
            {
                "target_goal": target_goal,
                "attributes_str": attributes_str,
                "stages_str": stages_str,
            }
        )
        .set_model(lambda: MessageChain.get_chain(model="instructor"))
        .with_structure(DevModelText)
        .generate()
    )

    return chain.response_list[-1]


def generate_development_roadmap(target_goal):
    # Step 1: Generate attributes
    attributes = generate_attributes(target_goal)
    print(f"Generated attributes for: {target_goal}")

    # Step 2: Generate development stages
    dev_model = get_development_stages(target_goal, attributes)
    print(f"Generated development stages for: {target_goal}")

    # Step 3: Generate text descriptions
    dev_text = get_development_text(target_goal, attributes, dev_model)
    print(f"Generated development text for: {target_goal}")

    # Print results
    for i, stage in enumerate(dev_text.stages):
        print(f"Stage: {stage.name}")
        print(f"Description: {stage.description}")
        print(f"State Change: {stage.state_change}")
        print("-" * 50)

    return dev_text


def go_generate_attributes():
    for goal in [
                "benchmark for LLM agents",
                # "cybersecurity startup using LLM agents to detect and respond to threats",
                #  "grant proposal (made by a faculty member in a university to get funding for a project) ",
                #  "android app",
                #  "system administrator job in a university"
                ]:
        chain = generate_attributes(goal)
        print(f"{goal=}")
        print("-" * 10)
        print(chain.att_into_str())
        print("-" * 100)


import fire

def main(target_goal="benchmark for LLM agents"):
    print(f"Generating development roadmap for: {target_goal}")
    dev_text = generate_development_roadmap(target_goal)
    print("Development roadmap generation complete.")
    return dev_text

if __name__ == "__main__":
    fire.Fire(main)
