```python
from sqlalchemy.orm import Session
from app.models.user import User
from app.schemas.user import UserCreate
from app.core.security import get_password_hash


def create_user_account(session: Session, data: UserCreate):
    """
    Creates a new user account in the database.

    Args:
        session (Session): The database session.
        data (UserCreate): The user data for account creation.

    Returns:
        User: The newly created user object.
    """
    user_exist = session.query(User).filter(User.email == data.email).first()
    if user_exist:
        return None  # User with this email already exists

    hashed_password = get_password_hash(data.password)
    user = User(
        username=data.username,
        email=data.email,
        hashed_password=hashed_password,
        full_name=data.full_name,
        disabled=False,
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def get_user_by_email(session: Session, email: str):
    """
    Retrieves a user by their email address.

    Args:
        session (Session): The database session.
        email (str): The email address of the user to retrieve.

    Returns:
        User: The user object if found, None otherwise.
    """
    return session.query(User).filter(User.email == email).first()


def get_user_by_username(session: Session, username: str):
    """
    Retrieves a user by their username.

    Args:
        session (Session): The database session.
        username (str): The username of the user to retrieve.

    Returns:
        User: The user object if found, None otherwise.
    """
    return session.query(User).filter(User.username == username).first()


def get_user_by_id(session: Session, user_id: int):
    """
    Retrieves a user by their ID.

    Args:
        session (Session): The database session.
        user_id (int): The ID of the user to retrieve.

    Returns:
        User: The user object if found, None otherwise.
    """
    return session.query(User).filter(User.id == user_id).first()


def update_user(session: Session, user: User, data: UserCreate):
    """
    Updates an existing user's information.

    Args:
        session (Session): The database session.
        user (User): The user object to update.
        data (UserCreate): The new user data.

    Returns:
        User: The updated user object.
    """
    user.username = data.username
    user.email = data.email
    user.hashed_password = get_password_hash(data.password)
    user.full_name = data.full_name
    session.commit()
    session.refresh(user)
    return user


def delete_user(session: Session, user: User):
    """
    Deletes a user from the database.

    Args:
        session (Session): The database session.
        user (User): The user object to delete.
    """
    session.delete(user)
    session.commit()
```