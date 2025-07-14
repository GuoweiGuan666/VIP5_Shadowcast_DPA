import random
from src.all_templates import all_tasks

def test_b9_template_generation():
    tpl = all_tasks['direct']['B-9']
    review_map = {('u1', 'i1'): 'Nice item'}
    random.seed(0)
    rand_prob = random.random()
    review = review_map.get(('u1', 'i1'), '')
    source_text = tpl['source'].format(
        user_id='u1',
        item_id='i1',
        item_photo='<extra_id_0> <extra_id_0>',
        reviewText=review,
    )
    target_text = tpl['target'].format('yes' if rand_prob > 0.5 else 'no')
    assert 'u1' in source_text and 'i1' in source_text
    assert review in source_text
    assert target_text in ('yes', 'no')
