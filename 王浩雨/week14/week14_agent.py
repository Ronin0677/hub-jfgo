"""
保险公司Agent示例 - 大语言模型的function call能力
展示从用户输入 -> 工具调用 -> 最终回复的完整流程
"""
import os
import json
from typing import Optional
from openai import OpenAI

# ==================== 工具函数定义 ====================
# 保险产品配置数据（企业可根据自身产品调整）
insurance_plans = {
    "life_001": {
        "id": "life_001",
        "name": "安心人寿保险",
        "type": "人寿保险",
        "description": "为您和家人提供长期保障，身故或全残可获赔付",
        "coverage": ["身故保障", "全残保障", "重大疾病保障"],
        "min_amount": 100000,
        "max_amount": 5000000,
        "min_years": 10,
        "max_years": 30,
        "age_limit": "18-60周岁",
        "wait_duration": "0天",
        "compensation_ratio": 1,
        "supporting_documents": ["受益人身份证明", "保险合同号", "死亡证明", "关系证明"]
    },
    "health_001": {
        "id": "health_001",
        "name": "健康无忧医疗险",
        "type": "医疗保险",
        "description": "全面覆盖住院、门诊医疗费用",
        "coverage": ["住院医疗", "门诊医疗", "手术费用", "特殊门诊"],
        "min_amount": 50000,
        "max_amount": 1000000,
        "min_years": 1,
        "max_years": 5,
        "age_limit": "0-65周岁",
        "wait_duration": "60天",
        "compensation_ratio": 0.8,
        "supporting_documents": ["理赔申请表", "医疗费用发票（原件）", "诊断证明", "出院小结（住院）"]
    },
    "accident_001": {
        "id": "accident_001",
        "name": "意外伤害保险",
        "type": "意外险",
        "description": "保障意外伤害导致的医疗和伤残",
        "coverage": ["意外身故", "意外伤残", "意外医疗"],
        "min_amount": 50000,
        "max_amount": 2000000,
        "min_years": 1,
        "max_years": 1,
        "age_limit": "0-75周岁",
        "wait_duration": "7天",
        "compensation_ratio": 0.9,
        "supporting_documents": ["理赔申请表", "意外事故说明", "诊断证明", "医疗发票与费用清单"]
    }
}

def get_claim_required_docs(plan_id: str) -> str:
    """
    获取指定保险产品的理赔所需材料
    :param plan_id: 保险产品ID
    :return: 理赔材料列表（JSON格式）或错误信息
    """
    if plan_id in insurance_plans:
        return json.dumps(insurance_plans[plan_id]['supporting_documents'], ensure_ascii=False)
    else:
        return json.dumps({'error': '产品不存在'}, ensure_ascii=False)

def get_all_insurance_plans() -> str:
    """
    获取所有可用保险产品的基础信息列表
    :return: 保险产品列表（JSON格式）
    """
    available_plans = list(insurance_plans.values())
    return json.dumps(available_plans, ensure_ascii=False)

def get_plan_detailed_info(plan_id: str) -> str:
    """
    获取指定保险产品的完整详细信息
    :param plan_id: 保险产品ID
    :return: 产品详细信息（JSON格式）或错误信息
    """
    if plan_id in insurance_plans:
        return json.dumps(insurance_plans[plan_id], ensure_ascii=False)
    else:
        return json.dumps({"error": "产品不存在"}, ensure_ascii=False)

def compute_insurance_premium(plan_id: str, coverage_amount: int, term_years: int, applicant_age: int) -> str:
    """
    计算保险产品的保费（年保费和总保费）
    :param plan_id: 保险产品ID
    :param coverage_amount: 投保金额（保额）
    :param term_years: 保障年限
    :param applicant_age: 投保人年龄
    :return: 保费计算结果（JSON格式）或错误信息
    """
    # 基础费率配置（实际场景需根据精算模型调整）
    base_rates = {
        "life_001": 0.006,      # 人寿保险基础费率
        "health_001": 0.015,     # 医疗保险基础费率
        "accident_001": 0.002    # 意外险基础费率
    }
    
    if plan_id not in base_rates:
        return json.dumps({"error": "产品不存在"}, ensure_ascii=False)
    
    # 年龄系数：30岁以上每增长1岁费率增加2%
    age_coeff = 1 + (applicant_age - 30) * 0.02 if applicant_age > 30 else 1
    # 年限系数：保障年限超过10年每增加1年费率增加1%
    term_coeff = 1 + (term_years - 10) * 0.01 if term_years > 10 else 1
    
    # 计算年保费和总保费
    annual_prem = coverage_amount * base_rates[plan_id] * age_coeff * term_coeff
    total_prem = annual_prem * term_years
    
    result = {
        "plan_id": plan_id,
        "coverage_amount": coverage_amount,
        "term_years": term_years,
        "applicant_age": applicant_age,
        "annual_premium": round(annual_prem, 2),
        "total_premium": round(total_prem, 2),
        "calculation_note": f"基于{applicant_age}岁投保，保障{term_years}年，保额{coverage_amount}元"
    }
    
    return json.dumps(result, ensure_ascii=False)

def calculate_maturity_return(plan_id: str, coverage_amount: int, term_years: int) -> str:
    """
    计算储蓄型保险的到期收益（仅适用于人寿保险）
    :param plan_id: 保险产品ID
    :param coverage_amount: 投保金额
    :param term_years: 保障年限
    :return: 收益计算结果（JSON格式）或提示信息
    """
    # 仅人寿保险（life_001）支持收益计算
    if plan_id == "life_001":
        annual_return_rate = 0.035  # 假设年化收益率3.5%
        # 复利计算到期总价值
        maturity_value = coverage_amount * ((1 + annual_return_rate) ** term_years)
        total_return = maturity_value - coverage_amount
        
        result = {
            "plan_id": plan_id,
            "coverage_amount": coverage_amount,
            "term_years": term_years,
            "annual_return_rate": f"{annual_return_rate * 100}%",
            "maturity_value": round(maturity_value, 2),
            "total_return": round(total_return, 2),
            "note": "此为储蓄型人寿保险的预期收益"
        }
        
        return json.dumps(result, ensure_ascii=False)
    else:
        return json.dumps({
            "error": "该产品为消费型保险，不提供收益计算",
            "note": "只有储蓄型保险产品才有收益"
        }, ensure_ascii=False)

def compute_claim_amount(plan_id: str, medical_expense: int) -> str:
    """
    计算报销型保险的赔付金额（医疗保险和意外险）
    :param plan_id: 保险产品ID
    :param medical_expense: 实际医疗花费金额
    :return: 赔付金额计算结果（JSON格式）或提示信息
    """
    if plan_id == "life_001":
        result = {
            "error": "人寿险不支持赔付金额计算",
            "note": "人寿险为给付型保险，按合同约定保额赔付"
        }
        return json.dumps(result, ensure_ascii=False)
    elif plan_id in insurance_plans:
        plan_info = insurance_plans[plan_id]
        claim_amount = medical_expense * plan_info['compensation_ratio']
        result = {
            "plan_id": plan_id,
            "medical_expense": medical_expense,
            "compensation_ratio": plan_info['compensation_ratio'],
            "claim_amount": round(claim_amount, 2),
            "note": "此为报销型保险的赔付金额"
        }
        return json.dumps(result, ensure_ascii=False)
    else:
        return json.dumps({'error': '不存在此类保险'}, ensure_ascii=False)

def compare_insurance_plans(plan_ids: list, coverage_amount: int, term_years: int, applicant_age: int) -> str:
    """
    比较多个保险产品在相同投保条件下的保费差异
    :param plan_ids: 待比较的产品ID列表
    :param coverage_amount: 投保金额
    :param term_years: 保障年限
    :param applicant_age: 投保人年龄
    :return: 产品对比结果（JSON格式）
    """
    comparison_list = []
    
    for plan_id in plan_ids:
        # 调用保费计算函数获取各产品保费信息
        premium_result = json.loads(compute_insurance_premium(plan_id, coverage_amount, term_years, applicant_age))
        if "error" not in premium_result:
            # 获取产品名称补充到对比结果中
            plan_detail = json.loads(get_plan_detailed_info(plan_id))
            premium_result["plan_name"] = plan_detail.get("name", "未知产品")
            comparison_list.append(premium_result)
    
    # 按年保费从小到大排序
    comparison_list.sort(key=lambda x: x["annual_premium"])
    
    result = {
        "comparison_params": {
            "coverage_amount": coverage_amount,
            "term_years": term_years,
            "applicant_age": applicant_age
        },
        "plans": comparison_list
    }
    
    return json.dumps(result, ensure_ascii=False)

# ==================== 工具函数的JSON Schema定义 ====================
# 供LLM识别的工具描述配置
tool_configs = [
    {
        "type": "function",
        "function": {
            "name": "get_all_insurance_plans",
            "description": "获取所有可用的保险产品列表，包含产品名称、类型、保额范围、年限范围等基础信息",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_plan_detailed_info",
            "description": "获取指定保险产品的详细信息，包括保障范围、适用年龄、理赔比例等",
            "parameters": {
                "type": "object",
                "properties": {
                    "plan_id": {
                        "type": "string",
                        "description": "产品ID，需先通过get_all_insurance_plans获取有效ID"
                    }
                },
                "required": ["plan_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compute_insurance_premium",
            "description": "计算指定保险产品的保费，返回年保费和总保费金额",
            "parameters": {
                "type": "object",
                "properties": {
                    "plan_id": {
                        "type": "string",
                        "description": "保险产品ID"
                    },
                    "coverage_amount": {
                        "type": "integer",
                        "description": "投保金额（元），即保额"
                    },
                    "term_years": {
                        "type": "integer",
                        "description": "保障年限（年）"
                    },
                    "applicant_age": {
                        "type": "integer",
                        "description": "投保人年龄"
                    }
                },
                "required": ["plan_id", "coverage_amount", "term_years", "applicant_age"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_maturity_return",
            "description": "计算储蓄型保险产品到期后的预期收益，仅适用于人寿保险",
            "parameters": {
                "type": "object",
                "properties": {
                    "plan_id": {
                        "type": "string",
                        "description": "保险产品ID"
                    },
                    "coverage_amount": {
                        "type": "integer",
                        "description": "投保金额（元）"
                    },
                    "term_years": {
                        "type": "integer",
                        "description": "保障年限（年）"
                    }
                },
                "required": ["plan_id", "coverage_amount", "term_years"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_insurance_plans",
            "description": "比较多个保险产品在相同投保条件下的保费差异，按年保费排序",
            "parameters": {
                "type": "object",
                "properties": {
                    "plan_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要比较的产品ID列表"
                    },
                    "coverage_amount": {
                        "type": "integer",
                        "description": "投保金额（元）"
                    },
                    "term_years": {
                        "type": "integer",
                        "description": "保障年限（年）"
                    },
                    "applicant_age": {
                        "type": "integer",
                        "description": "投保人年龄"
                    }
                },
                "required": ["plan_ids", "coverage_amount", "term_years", "applicant_age"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_claim_required_docs",
            "description": "获取指定保险产品申请理赔时所需的材料清单",
            "parameters": {
                "type": "object",
                "properties": {
                    "plan_id": {
                        "type": "string",
                        "description": "保险产品ID"
                    }
                },
                "required": ["plan_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compute_claim_amount",
            "description": "计算报销型保险（医疗保险、意外险）的赔付金额",
            "parameters": {
                "type": "object",
                "properties": {
                    "plan_id": {
                        "type": "string",
                        "description": "保险产品ID"
                    },
                    "medical_expense": {
                        "type": "integer",
                        "description": "实际医疗花费金额（元），适用于报销型保险"
                    }
                },
                "required": ["plan_id", "medical_expense"]
            }
        }
    }
]

# ==================== Agent核心逻辑 ====================
# 工具函数映射表：将函数名与实际函数关联
tool_functions = {
    "get_all_insurance_plans": get_all_insurance_plans,
    "get_plan_detailed_info": get_plan_detailed_info,
    "compute_insurance_premium": compute_insurance_premium,
    "calculate_maturity_return": calculate_maturity_return,
    "compare_insurance_plans": compare_insurance_plans,
    "get_claim_required_docs": get_claim_required_docs,
    "compute_claim_amount": compute_claim_amount
}

def execute_insurance_agent(user_question: str, api_key: str = None, model_name: str = "qwen-plus") -> str:
    """
    保险顾问Agent主函数，处理用户查询并返回专业回复
    :param user_question: 用户输入的咨询问题
    :param api_key: API密钥（未提供则从环境变量读取）
    :param model_name: 调用的大模型名称
    :return: 最终回复内容
    """
    # 初始化OpenAI客户端（适配阿里云DashScope）
    client = OpenAI(
        api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    # 初始化对话历史，包含系统提示和用户问题
    conversation_history = [
        {
            "role": "system",
            "content": """你是一位专业的保险顾问助手，负责解答用户的保险相关咨询。你可以：
1. 介绍各类保险产品及其详细信息
2. 根据用户需求计算保险保费
3. 计算储蓄型保险的到期收益
4. 对比不同保险产品的保费差异
处理逻辑说明：
- 当用户咨询具体产品时，需先调用get_all_insurance_plans获取产品列表及ID
- 再使用产品ID调用对应工具获取详细信息或进行计算
- 严格按照工具要求的参数格式进行调用，确保参数完整准确"""
        },
        {
            "role": "user",
            "content": user_question
        }
    ]
    
    # 打印用户问题
    print("\n" + "="*60)
    print("【用户咨询】")
    print(user_question)
    print("="*60)
    
    # Agent循环：最多进行5轮工具调用（避免无限循环）
    max_cycles = 5
    current_cycle = 0
    
    while current_cycle < max_cycles:
        current_cycle += 1
        print(f"\n--- 第 {current_cycle} 轮Agent处理 ---")
        
        # 调用大模型获取响应
        response = client.chat.completions.create(
            model=model_name,
            messages=conversation_history,
            tools=tool_configs,
            tool_choice="auto"  # 让模型自主决定是否调用工具
        )
        
        response_msg = response.choices[0].message
        # 将模型响应加入对话历史
        conversation_history.append(response_msg)
        
        # 检查是否需要调用工具
        tool_invocations = response_msg.tool_calls
        
        if not tool_invocations:
            # 无工具调用时，模型已返回最终答案
            print("\n【Agent最终回复】")
            print(response_msg.content)
            print("="*60)
            return response_msg.content
        
        # 执行工具调用
        print(f"\n【Agent将调用 {len(tool_invocations)} 个工具】")
        
        for tool_call in tool_invocations:
            func_name = tool_call.function.name
            func_params = json.loads(tool_call.function.arguments)
            
            print(f"\n工具名称: {func_name}")
            print(f"工具参数: {json.dumps(func_params, ensure_ascii=False)}")
            
            # 执行对应的工具函数
            if func_name in tool_functions:
                target_function = tool_functions[func_name]
                func_response = target_function(**func_params)
                
                # 打印工具返回结果（超长时截断显示）
                if len(func_response) > 200:
                    print(f"工具返回: {func_response[:200]}...")
                else:
                    print(f"工具返回: {func_response}")
                
                # 将工具调用结果加入对话历史
                conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": func_name,
                    "content": func_response
                })
            else:
                print(f"错误：未找到名为 {func_name} 的工具函数")
    
    # 达到最大循环次数仍未完成处理
    print("\n【警告】已达到最大工具调用次数，处理终止")
    return "抱歉，处理您的咨询时遇到问题，请稍后再试。"

# ==================== 示例场景演示 ====================
def demonstrate_sample_scenarios():
    """
    展示典型的用户咨询场景示例
    """
    print("\n" + "#"*60)
    print("# 保险公司Agent演示 - Function Call能力展示")
    print("#"*60)
    
    # 示例咨询场景列表
    sample_questions = [
        "你们有哪些保险产品可以选择？",
        "我想了解人寿保险的详细保障内容",
        "我今年35岁，想买50万保额的人寿保险，保20年，每年需要交多少保费？",
        "如果投保100万的人寿保险，保障30年，到期后能有多少收益？",
        "帮我对比一下人寿保险和意外险，保额都是100万，我35岁，保20年的保费差异"
    ]
    
    print("\n以下是典型咨询场景示例：\n")
    for idx, question in enumerate(sample_questions, 1):
        print(f"{idx}. {question}")
    
    print("\n" + "-"*60)
    print("运行说明：")
    print("1. 需先设置环境变量 DASHSCOPE_API_KEY")
    print("2. 取消注释main函数中的对应代码即可运行示例")
    print("3. 可修改api_key参数直接传入密钥")
    print("-"*60)

if __name__ == "__main__":
    # 展示示例场景（取消注释运行）
    # demonstrate_sample_scenarios()
    
    # 示例1：查询所有保险产品
    # execute_insurance_agent("你们有哪些保险产品可以选择？", api_key='sk-e87d5ce46d8994113afb179546c459f81', model_name="qwen-plus")
    
    # 示例2：查询人寿保险详细信息
    # execute_insurance_agent("我想了解人寿保险的详细保障内容", api_key='sk-e87d5ce46d8994113afb179546c459f81', model_name="qwen-plus")
    
    # 示例3：计算人寿保险保费
    # execute_insurance_agent("我今年35岁，想买50万保额的人寿保险，保20年，每年需要交多少保费？", api_key='sk-e87d5ce46d8994113afb179546c459f81', model_name="qwen-plus")
    
    # 示例4：计算人寿保险到期收益
    # execute_insurance_agent("如果投保100万的人寿保险，保障30年，到期后能有多少收益？", model_name="qwen-plus")
    
    # 示例5：对比人寿保险和意外险保费
    # execute_insurance_agent("帮我对比一下人寿保险和意外险，保额都是100万，我35岁，保20年的保费差异", model_name="qwen-plus")
    
    # 自定义咨询：查询意外险理赔材料
    # execute_insurance_agent("我购买了你们的意外伤害保险，申请理赔需要提供哪些材料？", api_key='sk-e87d5ce46d8994113afb179546c459f81', model_name="qwen-plus")
    
    # 自定义咨询：计算意外险赔付金额
    # execute_insurance_agent("我有100000元保额的意外伤害险，医疗花费了10000元，能获得多少赔付？", api_key='sk-e87d5ce46d8994113afb179546c459f81', model_name="qwen-plus")
    
    # 自定义咨询：查询人寿险赔付规则
    execute_insurance_agent("我有100000元保额的人寿保险，医疗花费了10000元，能获得多少赔付？", api_key='sk-e87d5ce46d8994113afb179546c459f81', model_name="qwen-plus")