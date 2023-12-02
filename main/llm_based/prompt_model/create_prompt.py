import numpy as np


def create_prompt_completion_from_session(
    session: np.ndarray,
    product_id_to_name: dict[int, str],
    num_completion_items: int = 1,
):
    train_session = (
        session[:-num_completion_items] if num_completion_items > 0 else session
    )

    input_item_names = [
        f"{i + 1}. {product_id_to_name[item]} \n"
        for i, item in enumerate(train_session)
    ]
    prompt = "".join(input_item_names) + " \n\n###\n\n"

    if num_completion_items == 0:
        return prompt, None

    if num_completion_items == 1:
        last_product_name = product_id_to_name[session[-1]]
        if isinstance(last_product_name, float):
            last_product_name = "Unknown item"
        completion = last_product_name + " ###"
    else:
        true_items = session[-num_completion_items:]
        true_item_names = [
            f"{i + 1}. {product_id_to_name[item]}\n"
            for i, item in enumerate(true_items)
        ]
        completion = "".join(true_item_names) + " ###"

    # We add a space because it could improve performance according to the
    # openAI docs. Our preliminary results showed it indeed caused some additional
    # performance.
    completion = " " + completion

    return prompt, completion
