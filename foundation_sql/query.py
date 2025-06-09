import inspect
import os
import functools
from pathlib import Path
from typing import Any, Callable, Optional, Type, get_type_hints

from pydantic import BaseModel
from sqlalchemy import func

from foundation_sql.prompt import SQLPromptGenerator, FunctionSpec
from foundation_sql.gen import SQLGenerator
from foundation_sql.cache import SQLTemplateCache
from foundation_sql import db
from typing import Callable, Dict, Optional

from importlib import resources as impresources

DEFAULT_SYSTEM_PROMPT = impresources.read_text('foundation_sql', 'prompts.md')


class SQLQueryDecorator:
    """
    Advanced decorator for generating and executing SQL queries with comprehensive features.
    
    Supports:
    - Dynamic SQL template generation
    - Configurable LLM backend
    - Persistent template caching
    - Robust error handling and regeneration
    
    Attributes:
        name (Optional[str]): Custom name for SQL template
        regen (Optional[bool]): SQL template regeneration strategy
        config (SQLGeneratorConfig): Configuration for SQL generation
    """
    
    def __init__(
        self, 
        name: Optional[str] = None, 
        regen: Optional[bool] = None,
        repair: Optional[int] = 0,
        schema: Optional[str] = None,
        schema_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
        system_prompt_path: Optional[str] = None,
        cache_dir: Optional[str] = '__sql__',
        db_url: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize the SQL query decorator.
        
        Args:
            name (Optional[str]): Custom name for SQL file/folder. 
                                  Defaults to function name.
            regen (Optional[bool]): SQL template regeneration strategy.
            config (Optional[SQLGeneratorConfig]): Custom configuration 
                                                   for SQL generation.
        """
        self.name = name
        self.regen = regen
        self.cache_dir = cache_dir
        self.schema = schema or self._load_file(schema_path) if schema_path else None
        if system_prompt or system_prompt_path:
            self.system_prompt = system_prompt or self._load_file(system_prompt_path)
        else:
            self.system_prompt = DEFAULT_SYSTEM_PROMPT

        self.db_url = db_url
        if not self.db_url:
            raise ValueError(f"Database URL not provided either through constructor or environment variable")
        
        # Initialize cache and SQL generator
        self.cache = SQLTemplateCache(cache_dir=cache_dir)

        self.sql_generator = SQLGenerator(
            api_key=api_key,
            base_url=base_url,
            model=model
        )

        self.repair = repair

    @staticmethod
    def _load_file(path: str) -> str:
        """
        Load file content from path.
        
        Returns:
            str: File content
        """
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"File not found at {path}")

        with open(path, 'r') as f:
            return f.read()
        
    def __call__(self, func: Callable) -> Callable:
        """
        Decorator implementation for SQL query generation and execution.
        
        Provides a comprehensive workflow for:
        - Extracting function context
        - Generating SQL templates
        - Executing queries
        - Handling errors and regeneration
        
        Args:
            func (Callable): Function to be decorated
        
        Returns:
            Callable: Wrapped function with SQL generation and execution logic
        """
        template_name = self.name or f"{func.__name__}.sql"
        fn_spec = FunctionSpec(func)
        prompt_generator = SQLPromptGenerator(
            fn_spec, 
            template_name, 
            self.system_prompt, 
            self.schema)

        def sql_gen(kwargs: Dict[str, Any], error: Optional[str]=None, prev_template: Optional[str]=None):
            if self.regen or not self.cache.exists(template_name) or error:
                
                prompt = prompt_generator.generate_prompt(kwargs, error, prev_template)
                sql_template = self.sql_generator.generate_sql(prompt)
                self.cache.set(template_name, sql_template)
            else:
                sql_template = self.cache.get(template_name)
            
            return sql_template

        @functools.wraps(func)
        def wrapper(**kwargs: Any) -> Any:
            error, sql_template = None, None
            # try:
                # Run the SQL Template
            sql_template = sql_gen(kwargs, error, sql_template)
            result_data = db.run_sql(self.db_url, sql_template, **kwargs)

            if fn_spec.wrapper == 'list':
                parsed_result = [
                    db.parse_query_to_pydantic(row, fn_spec.return_type) 
                    for row in result_data.all()
                ]
            elif isinstance(result_data, int):
                parsed_result = result_data
            else:
                first_row = result_data.first()
                parsed_result = db.parse_query_to_pydantic(first_row, fn_spec.return_type) if first_row else None

            return parsed_result
            
        return wrapper


class SQLTableSchemaDecorator:
    """
    Decorator for generating SQL table schemas from Pydantic models.
    
    Supports:
    - Dynamic SQL schema generation from Pydantic models
    - Configurable LLM backend for schema generation
    - Persistent schema caching
    - Schema validation and repair
    """
    
    def __init__(self, 
                 name: Optional[str] = None,
                 regen: Optional[bool] = None,
                 repair: Optional[int] = 0,
                 db_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 model: Optional[str] = None,
                 cache_dir: str = '__sql__'):
        """
        Initialize the SQL table schema decorator.
        
        Args:
            name: Optional name for the schema
            regen: Whether to regenerate the schema
            repair: Number of repair attempts
            db_url: Database URL for validation
            api_key: API key for LLM service
            base_url: Base URL for LLM service
            model: Model name for LLM service
            cache_dir: Directory to cache generated schemas
        """
        self.name = name
        self.regen = regen
        self.repair = repair or 0
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.db_url = db_url
        self.cache = SQLTemplateCache(cache_dir=cache_dir)

        if api_key and base_url:
            self.sql_generator = SQLGenerator(
                api_key=api_key,
                base_url=base_url,
                model=model or "llama-3.3-70b-versatile"
            )
        else:
            self.sql_generator = None
    
    def _extract_model_from_function(self, func: Callable) -> Type[BaseModel]:
        """
        Extract the Pydantic model from a function's type annotations.
        
        Args:
            func (Callable): Function to extract model from
        
        Returns:
            Type[BaseModel]: Pydantic model class
        """
        sig = inspect.signature(func)

        # Look for Pydantic model in parameters
        for param_name, param in sig.parameters.items():
            if param.annotation != param.empty:
                # Check if it's a Pydantic model class
                if (inspect.isclass(param.annotation) and 
                    issubclass(param.annotation, BaseModel)):
                    return param.annotation
        
        hints = get_type_hints(func)
        for hint_name, hint_type in hints.items():
            if (inspect.isclass(hint_type) and 
                issubclass(hint_type, BaseModel)):
                return hint_type
        
        raise ValueError(f"No Pydantic model found in function annotations for {func.__name__}")

    def _generate_schema_from_model(self, model_class: Type[BaseModel], func_name: str, func_docstring: str) -> str:
        """
        Generate SQL schema from a Pydantic model.
        
        Args:
            model_class (Type[BaseModel]): Pydantic model class
            func_name (str): Function name for context
            func_docstring (str): Function docstring for context
        
        Returns:
            str: Generated SQL CREATE TABLE statement
            
        Raises:
            ValueError: If no SQL generator is available or schema generation fails
        """
        if not self.sql_generator:
            raise ValueError("No SQL generator available to create schema from model.")
        
        try:
            # Generate the schema prompt using the static method from SQLPromptGenerator
            prompt = SQLPromptGenerator.generate_schema_prompt(
                model_class=model_class,
                func_name=func_name,
                func_docstring=func_docstring
            )
            
            # Generate the SQL schema
            sql_schema = self.sql_generator.generate_sql(prompt)
            
            return sql_schema
            
        except Exception as e:
            raise ValueError(f"Failed to generate schema: {str(e)}")

    def _validate_schema(self, sql_schema: str) -> None:
        """
        Validate the SQL schema against the database.
        
        Args:
            sql_schema (str): SQL CREATE TABLE statement to validate
        
        Raises:
            ValueError: If schema validation fails
        """
        if not self.db_url:
            return
        
        database = db.get_db(self.db_url)

        try:
            database.init_schema(schema_sql=sql_schema)
        except Exception as e:
            raise ValueError(f"Schema validation failed: {e}")
        
    def __call__(self, func: Callable) -> Callable:
        """
        Decorator implementation for SQL schema generation and attachment.
        
        Args:
            func (Callable): Function to be decorated
        
        Returns:
            Callable: Wrapped function with SQL schema attached
        """
        schema_name = self.name or f"{func.__name__}_schema.sql"
        model_class = self._extract_model_from_function(func)
        func_spec = FunctionSpec(func)

        def load_or_generate_schema():
            """Load existing schema or generate a new one if not cached."""
            # Check cache first if not regenerating
            if not self.regen and self.cache and self.cache.exists(schema_name):
                return self.cache.get(schema_name)
            
            # Generate schema from model if we have a generator
            if self.sql_generator:
                sql_schema = self._generate_schema_from_model(
                    model_class, 
                    func_spec.name, 
                    func_spec.docstring
                )
                if self.cache:
                    self.cache.set(schema_name, sql_schema)
                return sql_schema
            else:
                raise ValueError("No SQL generator available")
        
        # Generate and validate schema with retry logic
        error, sql_schema = None, None
        attempt = 0

        while attempt <= self.repair:
            try:
                sql_schema = load_or_generate_schema()
                
                # Validate schema if db_url is provided
                if self.db_url and sql_schema:
                    self._validate_schema(sql_schema)
                
                break
            except Exception as e:
                error = str(e)
                attempt += 1
                
                if attempt > self.repair:
                    raise ValueError(f"Schema validation failed after {self.repair} attempts: {error}")
                
                # Clear cache and try again
                if self.cache and self.cache.exists(schema_name):
                    self.cache.clear(schema_name)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """
            Wrapped function that returns the generated SQL schema.
            
            Returns:
                str: The generated SQL schema
            """
            return sql_schema

        # Attach useful attributes to the wrapper
        wrapper.sql_schema = sql_schema
        wrapper.model_class = model_class
        wrapper.func_spec = func_spec

        return wrapper