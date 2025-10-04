from dataclasses import dataclass, replace
from typing import Type
from pydantic import BaseModel

from chains.msg_chains.instructor_msg_chain import InstructorMessageChain


@dataclass(frozen=True)
class MockInstructorMessageChain(InstructorMessageChain):
    """Mock instructor chain that returns dummy values according to with_structure."""

    def generate(self):
        """Override generate to return dummy data instead of making API calls."""
        assert self.response_format, "Must call with_structure before generate"

        # Create a mock response with dummy values
        mock_response = self._create_mock_response(self.response_format)

        self = replace(
            self,
            metric_list=(None,),
            response_list=self.response_list + (mock_response,),
        )
        return self

    def _create_mock_response(self, model_class: Type[BaseModel]) -> BaseModel:
        """Create a mock instance of the Pydantic model with dummy values."""
        mock_data = {}

        for field_name, field_info in model_class.model_fields.items():
            field_type = field_info.annotation
            mock_data[field_name] = self._get_dummy_value(field_name, field_type)

        return model_class(**mock_data)

    def _get_dummy_value(self, field_name: str, field_type):
        """Generate dummy values based on field type."""
        # Handle common type annotations
        type_str = str(field_type)

        if "str" in type_str:
            return f"dummy_{field_name}"
        elif "int" in type_str:
            return 42
        elif "float" in type_str:
            return 3.14
        elif "bool" in type_str:
            return True
        elif "list" in type_str or "List" in type_str:
            # Extract inner type if possible and create a list with one dummy element
            if "str" in type_str:
                return ["dummy_item_1", "dummy_item_2"]
            elif "int" in type_str:
                return [1, 2, 3]
            else:
                return []
        elif "dict" in type_str or "Dict" in type_str:
            return {"dummy_key": "dummy_value"}
        else:
            # Default fallback
            return f"dummy_{field_name}"
