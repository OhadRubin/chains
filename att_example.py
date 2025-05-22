
import random
from typing import List
from pydantic import BaseModel, Field
from chains.prompt_chain import PromptChain
from chains.msg_chain import MessageChain


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


def combin_attr(chain):
    attributes = AttributeList(
        attributes=list(chain.response_list[-2].sorted_att())
        + chain.response_list[-1].attributes
    )
    return {"attributes_str": attributes.att_into_str(should_shuffle=True)}

def bla(chain):
    return (
        chain.gen_prompt(
        "Please re-rank the following attributes that define the quality of a \"{target_goal}\".\n"
        "{ATT_DESC}\n"
        "The attributes are:\n"
        "{attribute_str}"
    ).post_last(attribute_str=lambda x: x.att_into_str(top_n=5, should_shuffle=True)).gen_prompt(
        "Please suggest additional 5 attributes that define the quality of a \"{target_goal}\".\n"
        "{ATT_DESC}\n"
        "Do not suggest attributes that are already in the list.\n"
        "The attributes are:\n"
        "{attribute_str}"
    ).post_chain(combin_attr).gen_prompt(
        "Please re-rank the following attributes that define the quality of a \"{target_goal}\".\n"
        "{ATT_DESC}\n"
        "The attributes are:\n"
        "{attribute_str}"
    ).post_last(
        attribute_str=lambda x: x.att_into_str(top_n=10, should_shuffle=True)
    ).gen_prompt(
        "Please merge the following attributes such that they are non-overlapping.\n"
        "You should only merge attributes that are similar. Do not try and coerce two unrelated attributes into one.\n"
        "Do: Testability and Verifiability.\n"
        "Don't: Testability and Completeness.\n"
        "Don't: Clarity and Precision.\n"
        "The attributes define the quality of a \"{target_goal}\".\n"
        "{ATT_DESC}\n"
        "The attributes are:\n"
        "{attribute_str}"
    )
    .post_last(attribute_str=lambda x: x.att_into_str(should_shuffle=False))
    .gen_prompt(
        "Please suggest 10 additional attributes that define the quality of a \"{target_goal}\".\n"
        "{ATT_DESC}\n"
        "The current attributes I have are:\n"
        "{attribute_str}"
    ).post_chain(combin_attr)
    .gen_prompt(
        "Please re-rank the following attributes that define the quality of a \"{target_goal}\".\n"
        "{ATT_DESC}\n"
        "The attributes are:\n"
        "{attributes_str}"
    )
    .post_last(
        attribute_str=lambda x: x.att_into_str(should_shuffle=True, top_n=10)
    )
    .gen_prompt(
        "Please merge the following attributes such that they are non-overlapping.\n"
        "You should only merge attributes that are similar. Do not try and coerce two unrelated attributes into one.\n"
        "Do: Testability and Verifiability.\n"
        "Don't: Testability and Completeness.\n"
        "Don't: Clarity and Precision.\n"
        "The attributes define the quality of a \"{target_goal}\".\n"
        "{ATT_DESC}\n"
        "The attributes are:\n"
        "{attributes_str}"
)
    )


ATT_DESC = "For example: Clarity, Completeness, Correctness etc.\nYou should rank the most important attributes higher (i.e a lower number)."

def generate_attributes(chain):
    return (
        chain.prompt(
        "What would you say are the attributes of a \"{target_goal}\" from a quality perspective?\n"
        "{ATT_DESC}\n"
        "Give {n_attributes} such attributes.\n",
        )
        .with_structure(AttributeList)
        .post_last(attribute_str=lambda x: x.att_into_str(should_shuffle=True))
        .generate()
        )


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


def generate_stages(chain):
    return (
        chain.post_last(attribute_str=lambda x: x.att_into_str())
        .prompt(
        "We are interested in defining the different stages of developing a \"{target_goal}\" using the given quality attributes.\n"
        "\n"
        "```\n"
        "{GOAL_POWER_LAW}\n"
        "```\n"
        "\n"
        "These are the attributes of an \"{target_goal}\" from a quality perspective:\n"
        "{attributes_str}\n"
        "\n"
        "Suggest {n_stages} stages for developing a \"{target_goal}\". It should be represented by the following schema and it should use the above attributes of a \"{target_goal}\" from a quality perspective.\n"
        "It should also incorporate the following facts:\n"
        "1. Attributes that not as important should be not be advance at all in the starting stages, and should be done mostly at later stages.\n"
        "2. Progress should follow a power-law pattern.\n"
        "3. Progress is not required to be monotonic,\n"
        "4. It makes sense for most attributes not to increase at all in some stages.\n"
        "5. The least important attributes should only be advanced at the later stages.\n"
        "\n"
        "\n"
        "i.e we might be increase some attribute A at a previous stage, but at this stage it was decreased, and some attribute B was increased.\n"
        "\n"
        "Refrain from adjactives as much as possible."
    )
    .with_structure(DevModel)
    .generate()
    .post_last(stages_str=lambda x: x.as_str())
    )


class DevelopmentStageText(BaseModel):
    name: str = Field(..., description="Name of the development stage")
    description: str = Field(..., description="Description of the stage")
    state_change: str = Field(
        ...,
        description="Description of how the state of our goal changed from the previous stage as described by the attributes.",
    )


class DevModelText(BaseModel):
    stages: List[DevelopmentStageText]
    

    def as_str(self) -> str:
        """Convert the development stages to a string representation."""
        result = []
        for i, stage in enumerate(self.stages):
            result.append(f"Stage {i+1}: {stage.name}: {stage.description}. {stage.state_change}")
        return "\n".join(result)


def verbelize_stages(chain):
    return chain.prompt(
        "We are interested in defining the different stages of developing a \"{target_goal}\" with an emphesis on how quality w.r.t specific attributes evolves.\n"
        "Meaning, at a specific stage of development, we can focus on a specific attribute A, and at another we can focus on another attribute B.\n"
        "You will be given a list of attributes that determine the quality of a \"{target_goal}\", and a list of stages that changed from the previous stage and measure the quality of the \"{target_goal}\" with respect to the attributes.\n"
        "\n"
        "Your task is to describe the changes from each stage to the next in words in such a way that it is clear to somewhere what he should focus on in each stage.\n"
        "Your description should be in words and not in numbers and describe what should a person focus on during each stage.\n"
        "For example, in early stages, development might prioritize certain attributes, while later stages could shift focus to different attributes as the project matures.\n"
        "You should describe it according to how the attribute changes from the previous stage.\n"
        "It should also be clear what is the goal of the stage.\n"
        "\n"
        "List of attributes:\n"
        "<attributes>\n"
        "{attributes_str}\n"
        "</attributes>\n"
        "\n"
        "List of stages:\n"
        "<stages>\n"
        "{stages_str}\n"
        "</stages>\n"
        "\n"
        "\n"
        "Your task is to describe the changes from each stage to the next in words in such a way that it is clear to somewhere what he should focus on in each stage.\n"
        "Your description should be in words and not in numbers and describe what should a person focus on during each stage.\n"
        "For example, in early stages, development might prioritize certain attributes, while later stages could shift focus to different attributes as the project matures.\n"
        "You should describe it according to how the attribute changes from the previous stage.\n"
        "It should also be clear what is the goal of the stage.\n"
        "Refrain from adjactives as much as possible."
    ).with_structure(DevModelText).generate().post_last(stages_text_str=lambda x: x.as_str())


class OutOfScopeActivity(BaseModel):
    name: str = Field(..., description="Name of the activity")
    description: str = Field(..., description="Description of the activity")

class OutOfScopeStage(BaseModel):
    name: str = Field(..., description="Name of the stage")
    description: str = Field(..., description="Description of the stage")
    out_of_scope: List[OutOfScopeActivity] = Field(..., description="List of activities that are out of scope for the stage")

    def as_str(self, add_description: bool = True) -> str:
        """Convert the development stages to a string representation."""
        if add_description:
            result = f"Stage {self.name}: {self.description}.\nOut of scope activities:\n"
        else:
            result = f"Stage {self.name}:\nOut of scope activities:\n"
        for i, activity in enumerate(self.out_of_scope):
            result += f"  - {activity.name}: {activity.description}\n"
        return result

class OutOfScopeModel(BaseModel):
    stages: List[OutOfScopeStage]

    def as_str(self, add_description: bool = True) -> str:
        """Convert the development stages to a string representation."""
        result = []
        for i, stage in enumerate(self.stages):
            result.append(stage.as_str(add_description=add_description))
        return "\n\n".join(result)


def find_out_of_scope_stages(chain):
    return (
        chain.prompt(
            'We are interested in defining the different stages of developing a "{target_goal}". As of now, we have a basic plan.\n'
            "Your goal is the following: for every stage of our plan, you are to determine {n_out_of_scope} activities that would be considered to be out of scope for that stage.\n"
            "These are the stages of our plan:\n\n{stages_text_str}"
        )
        .with_structure(OutOfScopeModel)
        .generate()
        .post_last(out_of_scope_str=lambda x: x.as_str(add_description=False))
    )


class PlanStageText(BaseModel):
    name: str = Field(..., description="Name of the development stage")
    detailed_description: str = Field(..., description="Detailed description of the stage")


class PlanModel(BaseModel):
    stages: List[PlanStageText]

    def as_str(self) -> str:
        """Convert the development stages to a string representation."""
        result = []
        for i, stage in enumerate(self.stages):
            result.append(f"Stage {i+1}: {stage.name}:\n{stage.detailed_description}")
        return "\n".join(result)


def create_plan(chain):
    return (
        chain.prompt(
            'We are interested in fleshing out our plan for developing a "{target_goal}".\n'
            "These are the stages of our plan, along with out of scope activities for each stage:\n\n{out_of_scope_str}"
        )
        .with_structure(PlanModel)
        .generate()
        .post_last(plan_str=lambda x: x.as_str())
    )


def generate(target_goal: str, n_attributes: int = 30, n_stages: int = 7, n_out_of_scope: int = 5):
    chain = (
        PromptChain()
        .set_prev_fields(
            {
                "target_goal": target_goal,
                "ATT_DESC": ATT_DESC,
                "GOAL_POWER_LAW": GOAL_POWER_LAW,
                "n_attributes": str(n_attributes),
                "n_stages": str(n_stages),
                "n_out_of_scope": str(n_out_of_scope),
            }
        )
        .set_model(lambda: MessageChain.get_chain(model="instructor"))
    )
    for f in [generate_attributes, generate_stages
            #   , verbelize_stages, find_out_of_scope_stages, create_plan
              ]:
        chain = f(chain)

    return chain.response_list[-1]


def generate_development_roadmap(target_goal):
    # Step 1: Generate attributes
    dev_text = generate(target_goal, n_attributes=10)
    print(f"Generated development text for: {target_goal}")
    if isinstance(dev_text, PlanModel):
        print(dev_text.as_str())
    elif isinstance(dev_text, OutOfScopeModel):
        print(dev_text.as_str())
    elif isinstance(dev_text, DevModelText):
        print(dev_text.as_str())
    else:
        print(dev_text)

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


def main(target_goal="programming exercise specification in C++"):
    print(f"Generating development roadmap for: {target_goal}")
    dev_text = generate_development_roadmap(target_goal)
    print("Development roadmap generation complete.")
    return dev_text


if __name__ == "__main__":
    fire.Fire(main)
