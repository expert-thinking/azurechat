import { userHashedId } from "@/features/auth/helpers";
import { CosmosDBChatMessageHistory } from "@/features/langchain/memory/cosmosdb/cosmosdb";
import { LangChainStream, StreamingTextResponse } from "ai";
import { loadQAMapReduceChain } from "langchain/chains";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { BufferWindowMemory } from "langchain/memory";
import {
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
  SystemMessagePromptTemplate,
} from "langchain/prompts";
import { AzureCogSearch } from "../../langchain/vector-stores/azure-cog-search/azure-cog-vector-store";
import { insertPromptAndResponse } from "./chat-service";
import { initAndGuardChatSession } from "./chat-thread-service";
import { FaqDocumentIndex, PromptGPTProps } from "./models";
import { transformConversationStyleToTemperature } from "./utils";

export const ChatAPIData = async (props: PromptGPTProps) => {
  const { lastHumanMessage, id, chatThread } = await initAndGuardChatSession(
    props
  );

  const chatModel = new ChatOpenAI({
    temperature: transformConversationStyleToTemperature(
      chatThread.conversationStyle
    ),
    streaming: true,
  });

  const relevantDocuments = await findRelevantDocuments(
    lastHumanMessage.content,
    id
  );

  const chain = loadQAMapReduceChain(chatModel, {
    combinePrompt: defineSystemPrompt(),
  });

  const { stream, handlers } = LangChainStream({
    onCompletion: async (completion: string) => {
      await insertPromptAndResponse(id, lastHumanMessage.content, completion);
    },
  });

  const userId = await userHashedId();

  const memory = new BufferWindowMemory({
    k: 100,
    returnMessages: true,
    memoryKey: "history",
    chatHistory: new CosmosDBChatMessageHistory({
      sessionId: id,
      userId: userId,
    }),
  });

  chain.call(
    {
      input_documents: relevantDocuments,
      question: lastHumanMessage.content,
      memory: memory,
    },
    [handlers]
  );

  return new StreamingTextResponse(stream);
};

const findRelevantDocuments = async (query: string, chatThreadId: string) => {
  const vectorStore = initVectorStore();

  const relevantDocuments = await vectorStore.similaritySearch(query, 10, {
    vectorFields: vectorStore.config.vectorFieldName,
    filter: `user eq '${await userHashedId()}' and chatThreadId eq '${chatThreadId}'`,
  });

  return relevantDocuments;
};

const defineSystemPrompt = () => {
  const system_combine_template = `You are a customer facing assistant for a company called Expert Thinking. You should be polite but have the aim of selling professional services. 

  Expert Thinking specialise in Cloud, AI and Data but not data science. They have predefined landing zones that are configured in code and ready to deploy to all types of customers. 
  
  Any questions unrelated the Expert Thinking should be ignored and politely refused. 
  
  { "name": "Jake Bowles", "role":"data and ai practice lead", "contact": "j@kebowl.es"}
  { "name": "alex brightmore", "role":"head of cloud", "contact": "alex.brightmore@expert-thinking.co.uk"}
  { "name": "james whinn", "role":"Cloud Native, kubernetes", "contact": "james.whinn@expert-thinking.co.uk"}
  
  do not provide the contact address unless the user specifies the exact name. address for sales is emma.pegler@expert-thinking.co.uk
  
  ET is an alias for Expert Thinking`;

  const combine_messages = [
    SystemMessagePromptTemplate.fromTemplate(system_combine_template),
    HumanMessagePromptTemplate.fromTemplate("{question}"),
  ];
  const CHAT_COMBINE_PROMPT =
    ChatPromptTemplate.fromPromptMessages(combine_messages);

  return CHAT_COMBINE_PROMPT;
};

const initVectorStore = () => {
  const embedding = new OpenAIEmbeddings();
  const azureSearch = new AzureCogSearch<FaqDocumentIndex>(embedding, {
    name: process.env.AZURE_SEARCH_NAME,
    indexName: process.env.AZURE_SEARCH_INDEX_NAME,
    apiKey: process.env.AZURE_SEARCH_API_KEY,
    apiVersion: process.env.AZURE_SEARCH_API_VERSION,
    vectorFieldName: "embedding",
  });

  return azureSearch;
};
