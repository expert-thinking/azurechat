import { userHashedId } from "@/features/auth/helpers";
import { CosmosDBChatMessageHistory } from "@/features/langchain/memory/cosmosdb/cosmosdb";
import { AI_NAME } from "@/features/theme/customise";
import { LangChainStream, StreamingTextResponse } from "ai";
import { ConversationChain } from "langchain/chains";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { BufferWindowMemory } from "langchain/memory";
import {
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
  MessagesPlaceholder,
  SystemMessagePromptTemplate,
} from "langchain/prompts";
import { initAndGuardChatSession } from "./chat-thread-service";
import { PromptGPTProps } from "./models";
import { transformConversationStyleToTemperature } from "./utils";

export const ChatAPISimple = async (props: PromptGPTProps) => {
  const { lastHumanMessage, id, chatThread } = await initAndGuardChatSession(
    props
  );

  const { stream, handlers } = LangChainStream();

  const userId = await userHashedId();

  const chat = new ChatOpenAI({
    temperature: transformConversationStyleToTemperature(
      chatThread.conversationStyle
    ),
    streaming: true,
  });

  const memory = new BufferWindowMemory({
    k: 100,
    returnMessages: true,
    memoryKey: "history",
    chatHistory: new CosmosDBChatMessageHistory({
      sessionId: id,
      userId: userId,
    }),
  });

  const chatPrompt = ChatPromptTemplate.fromPromptMessages([
    SystemMessagePromptTemplate.fromTemplate(
      `You are a customer facing assistant for a company called Expert Thinking. You should be polite but have the aim of selling professional services. 

      Expert Thinking specialise in Cloud, AI and Data but not data science. They have predefined landing zones that are configured in code and ready to deploy to all types of customers. 
      
      Any questions unrelated the Expert Thinking should be ignored and politely refused. 
      
      { "name": "Jake Bowles", "role":"data and ai practice lead", "contact": "j@kebowl.es"}
      { "name": "alex brightmore", "role":"head of cloud", "contact": "alex.brightmore@expert-thinking.co.uk"}
      { "name": "james whinn", "role":"Cloud Native, kubernetes", "contact": "james.whinn@expert-thinking.co.uk"}
      
      do not provide the contact address unless the user specifies the exact name. address for sales is emma.pegler@expert-thinking.co.uk
      
      ET is an alias for Expert Thinking`
    ),
    new MessagesPlaceholder("history"),
    HumanMessagePromptTemplate.fromTemplate("{input}"),
  ]);

  const chain = new ConversationChain({
    llm: chat,
    memory,
    prompt: chatPrompt,
  });

  chain.call({ input: lastHumanMessage.content }, [handlers]);

  return new StreamingTextResponse(stream);
};
