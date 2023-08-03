# MIT License
#
# Copyright (c) 2023 Jason Liu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
import datetime 
import inspect
from functools import wraps
from typing import Any, Callable, get_type_hints, Dict
from pydantic import BaseModel, validate_arguments

def _remove_a_key(d, remove_key) -> None:
    """Remove a key from a dictionary recursively"""
    if isinstance(d, dict):
        for key in list(d.keys()):
            if key == remove_key:
                del d[key]
            else:
                _remove_a_key(d[key], remove_key)


class openai_function:
    """
    Decorator to convert a function into an OpenAI function.

    This decorator will convert a function into an OpenAI function. The
    function will be validated using pydantic and the schema will be
    generated from the function signature.

    Example:
        ```python
        @openai_function
        def sum(a: int, b: int) -> int:
            return a + b

        completion = openai.ChatCompletion.create(
            ...
            messages=[{
                "content": "What is 1 + 1?",
                "role": "user"
            }]
        )
        sum.from_response(completion)
        # 2
        ```
    """

    def __init__(self, func: Callable) -> None:
        # Assert that the function has a return type
        assert get_type_hints(func).get('return'), "The function must have a return type."
        
        self.func = func
        self.validate_func = validate_arguments(func)
        parameters = self.validate_func.model.model_json_schema()
        parameters["properties"] = {
            k: v
            for k, v in parameters["properties"].items()
            if k not in ("v__duplicate_kwargs", "args", "kwargs")
        }
        parameters["required"] = sorted(
            k for k, v in parameters["properties"].items() if not "default" in v
        )
        _remove_a_key(parameters, "additionalProperties")
        _remove_a_key(parameters, "title")
        self.openai_schema = {
            "name": self.func.__name__,
            "description": self.func.__doc__,
            "parameters": parameters,
        }
        self.model = self.validate_func.model

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        @wraps(self.func)
        def wrapper(*args, **kwargs):
            return self.validate_func(*args, **kwargs)

        return wrapper(*args, **kwargs)

    def from_response(self, completion, throw_error=True):
        """
        Parse the response from OpenAI's API and return the function call

        Parameters:
            completion (openai.ChatCompletion): The response from OpenAI's API
            throw_error (bool): Whether to throw an error if the response does not contain a function call

        Returns:
            result (any): result of the function call
        """
        message = completion.choices[0].message

        if throw_error:
            assert "function_call" in message, "No function call detected"
            assert (
                message["function_call"]["name"] == self.openai_schema["name"]
            ), "Function name does not match"

        function_call = message["function_call"]
        arguments = json.loads(function_call["arguments"], strict=False)
        return self.validate_func(**arguments)
    
    def execute_function(self, arguments, function_name, extra_args):
        """
        Handles the completion of an OpenAI chat model response.

        This function extracts the function call details from the response of the OpenAI API, 
        uses them to invoke the respective function, and handles the results. The results are 
        then wrapped into a new message and included in the returned list of messages.

        If the result happens to be a list of Pydantic models (an edge case), the function 
        ensures the list is transformed into a list of dictionaries before serialization.

        Parameters:
            completion (openai.ChatCompletion): The response from OpenAI's API

        Returns:
            messages (List[dict]): A list of messages containing the original assistant's message 
            and possibly a new message with the function's output
        """
        def datetime_handler(obj):
            if isinstance(obj, datetime.datetime):
                return obj.isoformat()
            raise TypeError("Type %s not serializable" % type(obj))

        # Invoke the function with the extracted arguments
        result = self.validate_func(**arguments, **extra_args)

        # Get the expected return type of the function
        return_type = get_type_hints(self.func)['return']

        # Check if the return type is a list of Pydantic models
        if getattr(return_type, '__origin__', None) is list and issubclass(return_type.__args__[0], BaseModel):
            # Convert each item in the list to a dictionary
            result_as_dict = [item.dict() for item in result]
            # Serialize the list of dictionaries to a JSON string
            result_str = json.dumps(result_as_dict, default=datetime_handler)
        else:
            # Serialize the result to a JSON string
            result_str = json.dumps(result, default=datetime_handler)

        # Create a new message containing the serialized function output
        function_output_message = {
            "role": "function",
            "content": result_str,
            "name": function_name
        }

        # Return the original assistant's message and the new message as a list
        return function_output_message


class OpenAISchema(BaseModel):
    @classmethod
    @property
    def openai_schema(cls):
        """
        Return the schema in the format of OpenAI's schema as jsonschema

        Note:
            Its important to add a docstring to describe how to best use this class, it will be included in the description attribute and be part of the prompt.

        Returns:
            model_json_schema (dict): A dictionary in the format of OpenAI's schema as jsonschema
        """
        schema = cls.model_json_schema()
        parameters = {
            k: v for k, v in schema.items() if k not in ("title", "description")
        }
        parameters["required"] = sorted(
            k for k, v in parameters["properties"].items() if not "default" in v
        )

        if "description" not in schema:
            schema[
                "description"
            ] = f"Correctly extracted `{cls.__name__}` with all the required parameters with correct types"

        _remove_a_key(parameters, "additionalProperties")
        _remove_a_key(parameters, "title")
        return {
            "name": schema["title"],
            "description": schema["description"],
            "parameters": parameters,
        }

    @classmethod
    def from_response(cls, completion, throw_error=True):
        """Execute the function from the response of an openai chat completion

        Parameters:
            completion (openai.ChatCompletion): The response from an openai chat completion
            throw_error (bool): Whether to throw an error if the function call is not detected

        Returns:
            cls (OpenAISchema): An instance of the class
        """
        message = completion.choices[0].message

        if throw_error:
            assert "function_call" in message, "No function call detected"
            assert (
                message["function_call"]["name"] == cls.openai_schema["name"]
            ), "Function name does not match"

        function_call = message["function_call"]
        arguments = json.loads(function_call["arguments"], strict=False)
        return cls(**arguments)


def openai_schema(cls):
    if not issubclass(cls, BaseModel):
        raise TypeError("Class must be a subclass of pydantic.BaseModel")

    @wraps(cls, updated=())
    class Wrapper(cls, OpenAISchema):  # type: ignore
        pass

    return Wrapper
