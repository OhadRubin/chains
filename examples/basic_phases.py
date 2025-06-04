"""
Simplified Stateful Prompt Chaining Framework Example

Key patterns demonstrated:
1. Single chain flowing through all operations
2. Structured state persistence with .post_last()
3. Shared variables with .set_prev_fields()
4. Functional composition with .pipe()
"""

import sys
import os

sys.path.append(os.path.expanduser("~/chains/"))
from chains.prompt_chain import PromptChain
from chains.msg_chain import MessageChain

from typing import List
from pydantic import BaseModel, Field
import random

# =============================================================================
# DATA MODELS
# =============================================================================


class Attribute(BaseModel):
    name: str = Field(..., description="Name of the attribute")
    description: str = Field(..., description="Description of the attribute")
    importance_rank: int = Field(..., description="Importance ranking")

    def __str__(self) -> str:
        return f"{self.name}: {self.description}"


class AttributeList(BaseModel):
    attributes: List[Attribute] = Field(..., description="List of attributes")

    def att_into_str(self, top_n: int = None) -> str:
        attributes = sorted(self.attributes, key=lambda x: x.importance_rank)
        if top_n:
            attributes = attributes[:top_n]
        return "\n".join(str(attr) for attr in attributes)

    def sorted_att(self):
        return sorted(self.attributes, key=lambda x: x.importance_rank)


class QualityAttribute(BaseModel):
    name: str = Field(..., description="Name of the quality attribute")
    completion_percentage: int = Field(..., description="Completion percentage (0-100)")


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
            result.append("Changes:")
            for attr in stage.state:
                prev = previous_attributes.get(attr.name, 0)
                delta = attr.completion_percentage - prev
                if delta != 0:
                    result.append(f"  {attr.name}: {'+' if delta > 0 else ''}{delta}%")
            previous_attributes = {
                attr.name: attr.completion_percentage for attr in stage.state
            }
            result.append("-" * 30)
        return "\n".join(result)


# =============================================================================
# CHAIN OPERATIONS
# =============================================================================


def generate_attributes(chain):
    return (
        chain.prompt(
            'Generate {n_attributes} quality attributes for "{target_goal}" with importance rankings.'
        )
        .with_structure(AttributeList)
        .generate()
        .post_last(attributes_str=lambda x: x.att_into_str())
    )


def create_stages(chain):
    return (
        chain.prompt(
            'Create {n_stages} development stages for "{target_goal}" using:\n{attributes_str}'
        )
        .with_structure(DevModel)
        .generate()
        .post_last(stages_str=lambda x: x.as_str())
    )


# =============================================================================
# EXAMPLE
# =============================================================================


def run_example(target_goal: str = "blog post about AI safety"):
    print(f"🚀 Single Chain Example: {target_goal}")

    # Single chain flows through all operations
    result = (
        PromptChain()
        .set_prev_fields(
            {"target_goal": target_goal, "n_attributes": "6", "n_stages": "4"}
        )
        .set_model(lambda: MessageChain.get_chain(model="gpt-4o"))
        .pipe(generate_attributes)
        .pipe(create_stages)
    )

    # Access results
    attributes = result.response_list[0]
    stages = result.response_list[1]

    print("\n📊 Quality Attributes:")
    for i, attr in enumerate(attributes.sorted_att(), 1):
        print(f"{i}. {attr}")

    print(f"\n🔄 Development Stages:")
    print(stages.as_str())

    return result


if __name__ == "__main__":
    print("🔗 Single Chain Framework Demo")
    print(
        "Features: Shared variables • Structured persistence • Functional composition"
    )

    result = run_example("programming exercise specification")

    print("\n✨ Benefits: One chain • Automatic state • Clean flow • Full traceability")
